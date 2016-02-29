//! This crate provides a stable, safe and scoped threadpool.
//!
//! It can be used to execute a number of short-lived jobs in parallel
//! without the need to respawn the underlying threads.
//!
//! Jobs are runnable by borrowing the pool for a given scope, during which
//! an arbitrary number of them can be executed. These jobs can access data of
//! any lifetime outside of the pools scope, which allows working on
//! non-`'static` references in parallel.
//!
//! For safety reasons, a panic inside a worker thread will not be isolated,
//! but rather propagate to the outside of the pool.
//!
//! # Examples:
//!
//! ```rust
//! extern crate scoped_threadpool;
//! use scoped_threadpool::Pool;
//!
//! fn main() {
//!     // Create a threadpool holding 4 threads
//!     let mut pool = Pool::new(4);
//!
//!     let mut vec = vec![0, 1, 2, 3, 4, 5, 6, 7];
//!
//!     // Use the threads as scoped threads that can
//!     // reference anything outside this closure
//!     pool.scoped(|scope| {
//!         // Create references to each element in the vector ...
//!         for e in &mut vec {
//!             // ... and add 1 to it in a seperate thread
//!             // (execute() is safe to call in nightly)
//!             scope.execute(move || {
//!                 *e += 1;
//!             });
//!         }
//!     });
//!     pool.join_and_stop();
//!
//!     assert_eq!(vec, vec![1, 2, 3, 4, 5, 6, 7, 8]);
//! }
//! ```

#![cfg_attr(all(feature="nightly", test), feature(test))]
#![cfg_attr(feature="nightly", feature(const_fn))]

#![warn(missing_docs)]

#[macro_use]
#[cfg(test)]
extern crate lazy_static;

use std::thread::{self, JoinHandle};
use std::sync::mpsc::{channel, Sender};
use std::sync::{Arc, Mutex};
use std::marker::PhantomData;
use std::mem;

enum Message {
    NewJob(Thunk<'static>),
    Join,
}

trait FnBox {
    fn call_box(self: Box<Self>);
}

impl<F: FnOnce()> FnBox for F {
    fn call_box(self: Box<F>) {
        (*self)()
    }
}

type Thunk<'a> = Box<FnBox + Send + 'a>;

impl Drop for Pool {
    fn drop(&mut self) {
        self.join_and_stop();
        self.job_sender = None;
    }
}

/// A threadpool that acts as a handle to a number
/// of threads spawned at construction.
pub struct Pool {
    job_sender: Option<Sender<Message>>,
    running: bool,
    thread_count: u32,
    threads: Vec<JoinHandle<()>>,
}

impl Pool {
    /// Construct a threadpool with the given number of threads.
    /// Minimum value is `1`.
    pub fn new(n: u32) -> Pool {
        assert!(n >= 1);

        let (job_sender, job_receiver) = channel();
        let job_receiver = Arc::new(Mutex::new(job_receiver));

        let mut threads = Vec::with_capacity(n as usize);

        // spawn n threads, put them in waiting mode
        for _ in 0..n {
            let job_receiver = job_receiver.clone();

            let thread = thread::spawn(move || {
                loop {
                    let message = {
                        // Only lock jobs for the time it takes
                        // to get a job, not run it.
                        let lock = job_receiver.lock().unwrap();
                        lock.recv()
                    };

                    match message {
                        Ok(Message::NewJob(job)) => {
                            job.call_box();
                        },
                        Ok(Message::Join) => break,
                        Err(..) => {
                            // The pool was dropped.
                            break
                        },
                    }
                }
            });

            threads.push(thread);
        }

        Pool {
            job_sender: Some(job_sender),
            running: true,
            thread_count: n,
            threads: threads,
        }
    }

    /// Borrows the pool and allows executing jobs on other
    /// threads during that scope via the argument of the closure.
    pub fn scoped<'pool, 'scope, F, R>(&'pool mut self, f: F) -> R
        where F: FnOnce(&mut Scope<'pool, 'scope>) -> R
    {
        let mut scope = Scope {
            pool: self,
            _marker: PhantomData,
        };
        f(&mut scope)
    }

    /// Returns the number of threads inside this pool.
    pub fn thread_count(&self) -> u32 {
        self.thread_count
    }

    /// Blocks until all currently queued jobs have run to completion.
    pub fn join_and_stop(&mut self) {
        if !self.running {
            return
        }

        for _ in 0..self.thread_count {
            self.job_sender.as_ref().unwrap().send(Message::Join).unwrap();
        }

        // Syncronize/Join with threads
        // This has to be a two step process
        // to make sure _all_ threads received _one_ Join message each.

        // This loop will block on every thread until it
        // received and reacted to its Join message.
        let threads = mem::replace(&mut self.threads, Vec::new());
        for join_handle in threads {
            join_handle.join().unwrap();
        }

        self.running = false
    }
}

/////////////////////////////////////////////////////////////////////////////

/// Handle to the scope during which the threadpool is borrowed.
pub struct Scope<'pool, 'scope> {
    pool: &'pool mut Pool,
    // The 'scope needs to be invariant... it seems?
    _marker: PhantomData<::std::cell::Cell<&'scope mut ()>>,
}

impl<'pool, 'scope> Scope<'pool, 'scope> {
    /// Execute a job on the threadpool.
    ///
    /// The body of the closure will be send to one of the
    /// internal threads, and this method itself will not wait
    /// for its completion.
    #[cfg(not(compiler_has_scoped_bugfix))]
    pub unsafe fn execute<F>(&self, f: F) where F: FnOnce() + Send + 'scope {
        self.execute_(f)
    }

    /// Execute a job on the threadpool.
    ///
    /// The body of the closure will be send to one of the
    /// internal threads, and this method itself will not wait
    /// for its completion.
    #[cfg(compiler_has_scoped_bugfix)]
    pub fn execute<F>(&self, f: F) where F: FnOnce() + Send + 'scope {
        self.execute_(f)
    }

    fn execute_<F>(&self, f: F) where F: FnOnce() + Send + 'scope {
        let b = unsafe {
            mem::transmute::<Thunk<'scope>, Thunk<'static>>(Box::new(f))
        };
        self.pool.job_sender.as_ref().unwrap().send(Message::NewJob(b)).unwrap();
    }

    /// Join all threads and stop pool
    ///
    /// This will wait for all threads to finish execution and return.
    /// It is a destructive action, and will render the pool unusable.
    pub fn join_and_stop(&mut self) {
        self.pool.join_and_stop();
    }
}

/////////////////////////////////////////////////////////////////////////////

#[cfg(test)]
mod tests {
    #![cfg_attr(feature="nightly", allow(unused_unsafe))]

    use std::sync;
    use std::thread;
    use std::time::Duration;

    use super::Pool;

    #[test]
    fn smoketest() {
        for i in 1..7 {
            let mut pool = Pool::new(4);

            let mut vec = vec![0, 1, 2, 3, 4];
            pool.scoped(|s| {
                for e in vec.iter_mut() {
                    s.execute(move || {
                        *e += i;
                    });
                }
            });
            pool.join_and_stop();

            let mut vec2 = vec![0, 1, 2, 3, 4];
            for e in vec2.iter_mut() {
                *e += i;
            }

            assert_eq!(vec, vec2);
        }
    }

    #[test]
    #[should_panic]
    fn thread_panic() {
        let mut pool = Pool::new(4);
        pool.scoped(|scoped| {
            scoped.execute(move || {
                panic!()
            });
        });
    }

    #[test]
    #[should_panic]
    fn scope_panic() {
        let mut pool = Pool::new(4);
        pool.scoped(|_scoped| {
            panic!()
        });
    }

    #[test]
    #[should_panic]
    fn pool_panic() {
        let _pool = Pool::new(4);
        panic!()
    }

    #[test]
    fn join_and_stop() {
        let mut pool = Pool::new(4);

        let (tx_, rx) = sync::mpsc::channel();

        pool.scoped(|mut scoped| {
            let tx = tx_.clone();
            scoped.execute(move || {
                thread::sleep(Duration::from_millis(1000));
                tx.send(2).unwrap();
            });

            let tx = tx_.clone();
            scoped.execute(move || {
                tx.send(1).unwrap();
            });

            scoped.join_and_stop();
        });

        pool.join_and_stop();
        pool.join_and_stop();

        assert_eq!(rx.iter().take(2).collect::<Vec<_>>(), vec![1, 2]);
    }

    #[test]
    #[cfg(compiler_has_scoped_bugfix)]
    fn safe_execute() {
        let mut pool = Pool::new(4);
        pool.scoped(|scoped| {
            scoped.execute(move || {
            });
        });
    }
}

#[cfg(all(test, feature="nightly"))]
mod benches {
    extern crate test;

    use self::test::{Bencher, black_box};
    use super::Pool;
    use std::sync::Mutex;

    // const MS_SLEEP_PER_OP: u32 = 1;

    lazy_static! {
        static ref POOL_1: Mutex<Pool> = Mutex::new(Pool::new(1));
        static ref POOL_2: Mutex<Pool> = Mutex::new(Pool::new(2));
        static ref POOL_3: Mutex<Pool> = Mutex::new(Pool::new(3));
        static ref POOL_4: Mutex<Pool> = Mutex::new(Pool::new(4));
        static ref POOL_5: Mutex<Pool> = Mutex::new(Pool::new(5));
        static ref POOL_8: Mutex<Pool> = Mutex::new(Pool::new(8));
    }

    fn fib(n: u64) -> u64 {
        let mut prev_prev: u64 = 1;
        let mut prev = 1;
        let mut current = 1;
        for _ in 2..(n+1) {
            current = prev_prev.wrapping_add(prev);
            prev_prev = prev;
            prev = current;
        }
        current
    }

    fn threads_interleaved_n(pool: &mut Pool)  {
        let size = 1024; // 1kiB

        let mut data = vec![1u8; size];
        pool.scoped(|s| {
            for e in data.iter_mut() {
                s.execute(move || {
                    *e += fib(black_box(1000 * (*e as u64))) as u8;
                    for i in 0..10000 { black_box(i); }
                    //thread::sleep_ms(MS_SLEEP_PER_OP);
                });
            }
        });
    }

    #[bench]
    fn threads_interleaved_1(b: &mut Bencher) {
        b.iter(|| threads_interleaved_n(&mut POOL_1.lock().unwrap()))
    }

    #[bench]
    fn threads_interleaved_2(b: &mut Bencher) {
        b.iter(|| threads_interleaved_n(&mut POOL_2.lock().unwrap()))
    }

    #[bench]
    fn threads_interleaved_4(b: &mut Bencher) {
        b.iter(|| threads_interleaved_n(&mut POOL_4.lock().unwrap()))
    }

    #[bench]
    fn threads_interleaved_8(b: &mut Bencher) {
        b.iter(|| threads_interleaved_n(&mut POOL_8.lock().unwrap()))
    }

    fn threads_chunked_n(pool: &mut Pool) {
        // Set this to 1GB and 40 to get good but slooow results
        let size = 1024 * 1024 * 10 / 4; // 10MiB
        let bb_repeat = 50;

        let n = pool.thread_count();
        let mut data = vec![0u32; size];
        pool.scoped(|s| {
            let l = (data.len() - 1) / n as usize + 1;
            for es in data.chunks_mut(l) {
                s.execute(move || {
                    if es.len() > 1 {
                        es[0] = 1;
                        es[1] = 1;
                        for i in 2..es.len() {
                            // Fibonnaci gets big fast,
                            // so just wrap around all the time
                            es[i] = black_box(es[i-1].wrapping_add(es[i-2]));
                            for i in 0..bb_repeat { black_box(i); }
                        }
                    }
                    //thread::sleep_ms(MS_SLEEP_PER_OP);
                });
            }
        });
    }

    #[bench]
    fn threads_chunked_1(b: &mut Bencher) {
        b.iter(|| threads_chunked_n(&mut POOL_1.lock().unwrap()))
    }

    #[bench]
    fn threads_chunked_2(b: &mut Bencher) {
        b.iter(|| threads_chunked_n(&mut POOL_2.lock().unwrap()))
    }

    #[bench]
    fn threads_chunked_3(b: &mut Bencher) {
        b.iter(|| threads_chunked_n(&mut POOL_3.lock().unwrap()))
    }

    #[bench]
    fn threads_chunked_4(b: &mut Bencher) {
        b.iter(|| threads_chunked_n(&mut POOL_4.lock().unwrap()))
    }

    #[bench]
    fn threads_chunked_5(b: &mut Bencher) {
        b.iter(|| threads_chunked_n(&mut POOL_5.lock().unwrap()))
    }

    #[bench]
    fn threads_chunked_8(b: &mut Bencher) {
        b.iter(|| threads_chunked_n(&mut POOL_8.lock().unwrap()))
    }
}

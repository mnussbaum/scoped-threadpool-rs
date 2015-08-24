var searchIndex = {};
searchIndex['scoped_threadpool'] = {"items":[[0,"","scoped_threadpool","This crate provides a stable, safe and scoped threadpool.",null,null],[3,"Pool","","A threadpool that acts as a handle to a number\nof threads spawned at construction.",null,null],[3,"Scope","","Handle to the scope during which the threadpool is borrowed.",null,null],[11,"drop","","",0,{"inputs":[{"name":"pool"}],"output":null}],[11,"new","","Construct a threadpool with the given number of threads.\nMinimum value is `1`.",0,{"inputs":[{"name":"pool"},{"name":"u32"}],"output":{"name":"pool"}}],[11,"scoped","","Borrows the pool and allows executing jobs on other\nthreads during that scope via the argument of the closure.",0,{"inputs":[{"name":"pool"},{"name":"f"}],"output":{"name":"r"}}],[11,"thread_count","","Returns the number of threads inside this pool.",0,{"inputs":[{"name":"pool"}],"output":{"name":"u32"}}],[11,"execute","","Execute a job on the threadpool.",1,{"inputs":[{"name":"scope"},{"name":"f"}],"output":null}],[11,"join_all","","Blocks until all currently queued jobs have run to completion.",1,{"inputs":[{"name":"scope"}],"output":null}],[11,"drop","","",1,{"inputs":[{"name":"scope"}],"output":null}]],"paths":[[3,"Pool"],[3,"Scope"]]};
searchIndex['rustc_version'] = {"items":[[0,"","rustc_version","Simple library for getting the version information of a `rustc`\ncompiler.",null,null],[3,"VersionMeta","","Rustc version plus metada like git short hash and build date.",null,null],[12,"semver","","Version of the compiler",0,null],[12,"git_short_hash","","Git short hash of the build of the compiler",0,null],[12,"date","","Build date of the compiler",0,null],[12,"channel","","Release channel of the compiler",0,null],[4,"Channel","","Release channel of the compiler.",null,null],[13,"Dev","","Development release channel",1,null],[13,"Nightly","","Nightly release channel",1,null],[13,"Beta","","Beta release channel",1,null],[13,"Stable","","Stable release channel",1,null],[5,"version","","Returns the `rustc` SemVer version.",null,{"inputs":[],"output":{"name":"version"}}],[5,"version_meta","","Returns the `rustc` SemVer version and additional metadata\nlike the git short hash and build date.",null,{"inputs":[],"output":{"name":"versionmeta"}}],[5,"version_matches","","Check wether the `rustc` version matches the given SemVer\nversion requirement.",null,{"inputs":[{"name":"str"}],"output":{"name":"bool"}}],[11,"hash","","",1,null],[11,"partial_cmp","","",1,{"inputs":[{"name":"channel"},{"name":"channel"}],"output":{"name":"option"}}],[11,"lt","","",1,{"inputs":[{"name":"channel"},{"name":"channel"}],"output":{"name":"bool"}}],[11,"le","","",1,{"inputs":[{"name":"channel"},{"name":"channel"}],"output":{"name":"bool"}}],[11,"gt","","",1,{"inputs":[{"name":"channel"},{"name":"channel"}],"output":{"name":"bool"}}],[11,"ge","","",1,{"inputs":[{"name":"channel"},{"name":"channel"}],"output":{"name":"bool"}}],[11,"cmp","","",1,{"inputs":[{"name":"channel"},{"name":"channel"}],"output":{"name":"ordering"}}],[11,"eq","","",1,{"inputs":[{"name":"channel"},{"name":"channel"}],"output":{"name":"bool"}}],[11,"ne","","",1,{"inputs":[{"name":"channel"},{"name":"channel"}],"output":{"name":"bool"}}],[11,"clone","","",1,{"inputs":[{"name":"channel"}],"output":{"name":"channel"}}],[11,"hash","","",0,null],[11,"partial_cmp","","",0,{"inputs":[{"name":"versionmeta"},{"name":"versionmeta"}],"output":{"name":"option"}}],[11,"lt","","",0,{"inputs":[{"name":"versionmeta"},{"name":"versionmeta"}],"output":{"name":"bool"}}],[11,"le","","",0,{"inputs":[{"name":"versionmeta"},{"name":"versionmeta"}],"output":{"name":"bool"}}],[11,"gt","","",0,{"inputs":[{"name":"versionmeta"},{"name":"versionmeta"}],"output":{"name":"bool"}}],[11,"ge","","",0,{"inputs":[{"name":"versionmeta"},{"name":"versionmeta"}],"output":{"name":"bool"}}],[11,"cmp","","",0,{"inputs":[{"name":"versionmeta"},{"name":"versionmeta"}],"output":{"name":"ordering"}}],[11,"eq","","",0,{"inputs":[{"name":"versionmeta"},{"name":"versionmeta"}],"output":{"name":"bool"}}],[11,"ne","","",0,{"inputs":[{"name":"versionmeta"},{"name":"versionmeta"}],"output":{"name":"bool"}}],[11,"clone","","",0,{"inputs":[{"name":"versionmeta"}],"output":{"name":"versionmeta"}}]],"paths":[[3,"VersionMeta"],[4,"Channel"]]};
searchIndex['semver'] = {"items":[[0,"","semver","Semantic version parsing and comparison.",null,null],[3,"Version","","Represents a version number conforming to the semantic versioning scheme.",null,null],[12,"major","","The major version, to be incremented on incompatible changes.",0,null],[12,"minor","","The minor version, to be incremented when functionality is added in a\nbackwards-compatible manner.",0,null],[12,"patch","","The patch version, to be incremented when backwards-compatible bug\nfixes are made.",0,null],[12,"pre","","The pre-release version identifier, if one exists.",0,null],[12,"build","","The build metadata, ignored when determining version precedence.",0,null],[3,"VersionReq","","A `VersionReq` is a struct containing a list of predicates that can apply to ranges of version\nnumbers. Matching operations can then be done with the `VersionReq` against a particular\nversion to see if it satisfies some or all of the constraints.",null,null],[4,"Identifier","","An identifier in the pre-release or build metadata.",null,null],[13,"Numeric","","An identifier that's solely numbers.",1,null],[13,"AlphaNumeric","","An identifier with letters and numbers.",1,null],[4,"ParseError","","A `ParseError` is returned as the `Err` side of a `Result` when a version is\nattempted to be parsed.",null,null],[13,"NonAsciiIdentifier","","All identifiers must be ASCII.",2,null],[13,"IncorrectParse","","The version was mis-parsed.",2,null],[13,"GenericFailure","","Any other failure.",2,null],[4,"ReqParseError","","A `ReqParseError` is returned from methods which parse a string into a `VersionReq`. Each\nenumeration is one of the possible errors that can occur.",null,null],[13,"InvalidVersionRequirement","","The given version requirement is invalid.",3,null],[13,"OpAlreadySet","","You have already provided an operation, such as `=`, `~`, or `^`. Only use one.",3,null],[13,"InvalidSigil","","The sigil you have written is not correct.",3,null],[13,"VersionComponentsMustBeNumeric","","All components of a version must be numeric.",3,null],[13,"MajorVersionRequired","","At least a major version is required.",3,null],[13,"UnimplementedVersionRequirement","","An unimplemented version requirement.",3,null],[11,"fmt","","",1,{"inputs":[{"name":"identifier"},{"name":"formatter"}],"output":{"name":"result"}}],[11,"hash","","",1,null],[11,"cmp","","",1,{"inputs":[{"name":"identifier"},{"name":"identifier"}],"output":{"name":"ordering"}}],[11,"partial_cmp","","",1,{"inputs":[{"name":"identifier"},{"name":"identifier"}],"output":{"name":"option"}}],[11,"lt","","",1,{"inputs":[{"name":"identifier"},{"name":"identifier"}],"output":{"name":"bool"}}],[11,"le","","",1,{"inputs":[{"name":"identifier"},{"name":"identifier"}],"output":{"name":"bool"}}],[11,"gt","","",1,{"inputs":[{"name":"identifier"},{"name":"identifier"}],"output":{"name":"bool"}}],[11,"ge","","",1,{"inputs":[{"name":"identifier"},{"name":"identifier"}],"output":{"name":"bool"}}],[11,"eq","","",1,{"inputs":[{"name":"identifier"},{"name":"identifier"}],"output":{"name":"bool"}}],[11,"ne","","",1,{"inputs":[{"name":"identifier"},{"name":"identifier"}],"output":{"name":"bool"}}],[11,"clone","","",1,{"inputs":[{"name":"identifier"}],"output":{"name":"identifier"}}],[11,"fmt","","",1,{"inputs":[{"name":"identifier"},{"name":"formatter"}],"output":{"name":"result"}}],[11,"fmt","","",0,{"inputs":[{"name":"version"},{"name":"formatter"}],"output":{"name":"result"}}],[11,"clone","","",0,{"inputs":[{"name":"version"}],"output":{"name":"version"}}],[11,"partial_cmp","","",2,{"inputs":[{"name":"parseerror"},{"name":"parseerror"}],"output":{"name":"option"}}],[11,"lt","","",2,{"inputs":[{"name":"parseerror"},{"name":"parseerror"}],"output":{"name":"bool"}}],[11,"le","","",2,{"inputs":[{"name":"parseerror"},{"name":"parseerror"}],"output":{"name":"bool"}}],[11,"gt","","",2,{"inputs":[{"name":"parseerror"},{"name":"parseerror"}],"output":{"name":"bool"}}],[11,"ge","","",2,{"inputs":[{"name":"parseerror"},{"name":"parseerror"}],"output":{"name":"bool"}}],[11,"fmt","","",2,{"inputs":[{"name":"parseerror"},{"name":"formatter"}],"output":{"name":"result"}}],[11,"eq","","",2,{"inputs":[{"name":"parseerror"},{"name":"parseerror"}],"output":{"name":"bool"}}],[11,"ne","","",2,{"inputs":[{"name":"parseerror"},{"name":"parseerror"}],"output":{"name":"bool"}}],[11,"clone","","",2,{"inputs":[{"name":"parseerror"}],"output":{"name":"parseerror"}}],[11,"parse","","Parse a string into a semver object.",0,{"inputs":[{"name":"version"},{"name":"str"}],"output":{"name":"result"}}],[11,"increment_patch","","Increments the patch number for this Version (Must be mutable)",0,{"inputs":[{"name":"version"}],"output":null}],[11,"increment_minor","","Increments the minor version number for this Version (Must be mutable)",0,{"inputs":[{"name":"version"}],"output":null}],[11,"increment_major","","Increments the major version number for this Version (Must be mutable)",0,{"inputs":[{"name":"version"}],"output":null}],[11,"fmt","","",0,{"inputs":[{"name":"version"},{"name":"formatter"}],"output":{"name":"result"}}],[11,"eq","","",0,{"inputs":[{"name":"version"},{"name":"version"}],"output":{"name":"bool"}}],[11,"partial_cmp","","",0,{"inputs":[{"name":"version"},{"name":"version"}],"output":{"name":"option"}}],[11,"cmp","","",0,{"inputs":[{"name":"version"},{"name":"version"}],"output":{"name":"ordering"}}],[11,"description","","",2,{"inputs":[{"name":"parseerror"}],"output":{"name":"str"}}],[11,"fmt","","",2,{"inputs":[{"name":"parseerror"},{"name":"formatter"}],"output":{"name":"result"}}],[11,"hash","","",0,{"inputs":[{"name":"version"},{"name":"h"}],"output":null}],[11,"fmt","","",4,{"inputs":[{"name":"versionreq"},{"name":"formatter"}],"output":{"name":"result"}}],[11,"clone","","",4,{"inputs":[{"name":"versionreq"}],"output":{"name":"versionreq"}}],[11,"eq","","",4,{"inputs":[{"name":"versionreq"},{"name":"versionreq"}],"output":{"name":"bool"}}],[11,"ne","","",4,{"inputs":[{"name":"versionreq"},{"name":"versionreq"}],"output":{"name":"bool"}}],[11,"eq","","",3,{"inputs":[{"name":"reqparseerror"},{"name":"reqparseerror"}],"output":{"name":"bool"}}],[11,"ne","","",3,{"inputs":[{"name":"reqparseerror"},{"name":"reqparseerror"}],"output":{"name":"bool"}}],[11,"fmt","","",3,{"inputs":[{"name":"reqparseerror"},{"name":"formatter"}],"output":{"name":"result"}}],[11,"clone","","",3,{"inputs":[{"name":"reqparseerror"}],"output":{"name":"reqparseerror"}}],[11,"fmt","","",3,{"inputs":[{"name":"reqparseerror"},{"name":"formatter"}],"output":{"name":"result"}}],[11,"description","","",3,{"inputs":[{"name":"reqparseerror"}],"output":{"name":"str"}}],[11,"any","","`any()` is a factory method which creates a `VersionReq` with no constraints. In other\nwords, any version will match against it.",4,{"inputs":[{"name":"versionreq"}],"output":{"name":"versionreq"}}],[11,"parse","","`parse()` is the main constructor of a `VersionReq`. It turns a string like `\"^1.2.3\"`\nand turns it into a `VersionReq` that matches that particular constraint.",4,{"inputs":[{"name":"versionreq"},{"name":"str"}],"output":{"name":"result"}}],[11,"exact","","`exact()` is a factory method which creates a `VersionReq` with one exact constraint.",4,{"inputs":[{"name":"versionreq"},{"name":"version"}],"output":{"name":"versionreq"}}],[11,"matches","","`matches()` matches a given `Version` against this `VersionReq`.",4,{"inputs":[{"name":"versionreq"},{"name":"version"}],"output":{"name":"bool"}}],[11,"fmt","","",4,{"inputs":[{"name":"versionreq"},{"name":"formatter"}],"output":{"name":"result"}}]],"paths":[[3,"Version"],[4,"Identifier"],[4,"ParseError"],[4,"ReqParseError"],[3,"VersionReq"]]};
initSearch(searchIndex);

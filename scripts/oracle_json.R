# Exact binary64 JSON numbers, including signed zero. Uses jsonlite only to
# quote strings; numeric serialization is not delegated to a rounding formatter.
encode_oracle_json <- function(value, depth=0L) {
  indent <- function(n) paste(rep("  ",n),collapse="")
  if(is.numeric(value)) {
    fields <- vapply(as.double(value),function(x) {
      if(is.na(x)) stop("non-finite NaN oracle number")
      if(is.infinite(x)) return(if(x>0) '"Inf"' else '"-Inf"')
      if(x==0 && (1/x)<0) return("-0.0")
      sprintf("%.17g",x)
    },character(1))
    if(length(fields)==1L && !inherits(value,"AsIs")) fields else paste0("[",paste(fields,collapse=", "),"]")
  } else if(is.null(value)) {
    "null"
  } else if(is.character(value) || is.logical(value)) {
    as.character(jsonlite::toJSON(value,auto_unbox=TRUE))
  } else if(is.list(value)) {
    keys<-names(value)
    parts<-vapply(seq_along(value),function(i) {
      prefix<-if(is.null(keys)) "" else paste0(as.character(jsonlite::toJSON(keys[[i]],auto_unbox=TRUE)),": ")
      paste0(indent(depth+1L),prefix,encode_oracle_json(value[[i]],depth+1L))
    },character(1))
    left<-if(is.null(keys)) "[" else "{";right<-if(is.null(keys)) "]" else "}"
    if(!length(parts)) return(paste0(left,right))
    paste0(left,"\n",paste(parts,collapse=",\n"),"\n",indent(depth),right)
  } else stop("unsupported oracle JSON value type")
}

verify_oracle_roundtrip <- function(original, parsed, path="root") {
  if(is.numeric(original)) {
    x<-as.double(original)
    y<-if(is.list(parsed)) vapply(parsed,as.double,numeric(1)) else as.double(parsed)
    if(!identical(writeBin(x,raw(),size=8,endian="little"),writeBin(y,raw(),size=8,endian="little"))) {
      stop("binary64 round-trip failed at ",path)
    }
  } else if(is.list(original)) {
    stopifnot(length(original)==length(parsed),identical(names(original),names(parsed)))
    for(i in seq_along(original)) verify_oracle_roundtrip(original[[i]],parsed[[i]],paste0(path,"/",i))
  } else if(is.null(original)) {
    stopifnot(is.null(parsed))
  }
  invisible(TRUE)
}

write_verified_oracle_json <- function(value,path) {
  dir.create(dirname(path),recursive=TRUE,showWarnings=FALSE)
  temporary<-tempfile(tmpdir=dirname(path));on.exit(unlink(temporary))
  writeLines(encode_oracle_json(value),temporary)
  verify_oracle_roundtrip(value,jsonlite::read_json(temporary,simplifyVector=FALSE))
  stopifnot(file.rename(temporary,path))
  invisible(path)
}

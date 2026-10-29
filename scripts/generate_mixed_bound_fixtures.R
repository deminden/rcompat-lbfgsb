#!/usr/bin/env Rscript
# Black-box optim calls on independently defined scalar quadratic/linear models.
args <- commandArgs(TRUE)
out <- if(length(args)) args[[1]] else "fixtures/mixed_bounds/cases.json"
stopifnot(requireNamespace("jsonlite", quietly=TRUE))
if(Sys.getenv("OPENBLAS_NUM_THREADS")!="1") stop("Start Rscript with OPENBLAS_NUM_THREADS=1")
cases <- list()
add <- function(n, kind, gradient, scale=1, maxit=100L, linear=FALSE, offset=0, start=NULL) {
  lower <- if(kind=="unbounded") rep(-Inf,n) else rep(-10,n)
  upper <- if(kind=="finite") rep(10,n) else rep(Inf,n)
  if(kind=="mixed" && n>1) upper[[n]]<-10
  initial <- if(is.null(start)) rep(c(4,-5),length.out=n) else rep(start,length.out=n)
  target <- rep(c(1,-2),length.out=n)
  weights <- rep(c(1,2),length.out=n)
  calls <- list()
  fn <- function(x) {
    calls[[length(calls)+1L]] <<- I(unname(x))
    value <- 0
    for(i in seq_len(n)) { d<-x[[i]]-target[[i]]; value<-value+if(linear) -x[[i]] else weights[[i]]*d*d }
    value*scale+offset
  }
  gr <- function(x) { g<-numeric(n); for(i in seq_len(n)) g[[i]]<-if(linear) -scale else (2*weights[[i]])*(x[[i]]-target[[i]])*scale;g }
  control<-list(maxit=maxit,factr=if(linear)0 else 1e7,pgtol=0,lmm=5)
  result<-optim(initial,fn,if(gradient)gr else NULL,method="L-BFGS-B",lower=lower,upper=upper,control=control)
  id<-paste(n,kind,gradient,format(scale),maxit,linear,format(offset),paste(initial,collapse=","),sep=":")
  cases[[length(cases)+1L]] <<- list(contract=if(offset!=0)"exact" else "numerical",id=id,n=n,kind=kind,gradient=gradient,scale=scale,linear=linear,offset=offset,
    initial=I(initial),target=I(target),weights=I(weights),lower=I(lower),upper=I(upper),control=control,
    result=list(par=I(unname(result$par)),value=result$value,counts=as.list(result$counts),convergence=result$convergence,message=result$message),
    first_calls=head(calls,80))
}
for(n in c(1L,2L,4L,13L)) for(kind in c("unbounded","mixed","finite")) for(gradient in c(FALSE,TRUE)) add(n,kind,gradient)
for(kind in c("unbounded","mixed")) for(gradient in c(FALSE,TRUE)) add(2L,kind,gradient,1e-8)
for(kind in c("unbounded","mixed","finite")) add(2L,kind,TRUE,maxit=0L)
add(2L,"unbounded",TRUE,linear=TRUE)
add(2L,"finite",TRUE,linear=TRUE)
add(1L,"finite",TRUE,scale=1e-20,linear=TRUE,offset=1,start=0)
add(1L,"unbounded",TRUE,scale=1e-20,linear=TRUE,offset=1,start=0)
add(1L,"finite",TRUE,scale=1e-20,linear=TRUE,offset=1,start=4)
add(1L,"unbounded",TRUE,scale=1e-20,linear=TRUE,offset=1,start=4)
script_path<-sub("^--file=", "", grep("^--file=",commandArgs(FALSE),value=TRUE)[[1]])
script_hash<-strsplit(system2("shasum",c("-a","256",shQuote(script_path)),stdout=TRUE),"[[:space:]]+")[[1]][[1]]
manifest<-list(generator_sha256=script_hash, r_version=R.version.string,platform=R.version$platform,blas=unname(extSoftVersion()[["BLAS"]]),
  threads=Sys.getenv("OPENBLAS_NUM_THREADS"),session=capture.output(sessionInfo()),generator="scripts/generate_mixed_bound_fixtures.R",cases=cases)
dir.create(dirname(out),recursive=TRUE,showWarnings=FALSE)
verify_roundtrip <- function(original, parsed) {
  if(is.numeric(original)) {
    x<-as.double(original);y<-as.double(unlist(parsed))
    stopifnot(identical(writeBin(x,raw(),size=8,endian="little"),writeBin(y,raw(),size=8,endian="little")))
  } else if(is.list(original)) {
    stopifnot(length(original)==length(parsed))
    for(i in seq_along(original)) verify_roundtrip(original[[i]],parsed[[i]])
  }
}
write_verified <- function() {
  temporary<-tempfile(tmpdir=dirname(out));on.exit(unlink(temporary))
  jsonlite::write_json(manifest,temporary,auto_unbox=TRUE,pretty=TRUE,digits=17)
  verify_roundtrip(manifest,jsonlite::read_json(temporary,simplifyVector=FALSE))
  stopifnot(file.rename(temporary,out))
}
write_verified()
cat(length(cases),"cases written to",out,"\n")

#!/usr/bin/env Rscript
# Replacement corpus from independent formulas, not the missing historical data.
args <- commandArgs(TRUE)
out <- if(length(args)) args[[1]] else "fixtures/synthetic_v1/cases.json"
stopifnot(requireNamespace("jsonlite", quietly=TRUE), as.character(getRversion())==Sys.getenv("ORACLE_R_VERSION","4.6.2"))
if(Sys.getenv("OPENBLAS_NUM_THREADS")!="1") stop("Start Rscript with OPENBLAS_NUM_THREADS=1")
cases <- list()
for(n in c(1L,2L,4L,9L,10L,11L,13L,32L)) {
 for(family in c("correlated_quadratic","ill_conditioned_quadratic","negative_binomial","double_well")) {
  for(route in c("unbounded","lower","mixed","box")) for(variant in 0:1) for(gradient in c(FALSE,TRUE)) {
   lower <- if(route=="unbounded") rep(-Inf,n) else rep(-1.5,n)
   upper <- if(route=="box") rep(1.5,n) else rep(Inf,n)
   if(route=="mixed") upper[seq(1L,n,by=2L)]<-1.5
   initial <- if(variant==0) 3-(seq_len(n)-1)%%3 else -2+0.125*((seq_len(n)-1)%%5)
   if(family=="double_well") initial<-rep(if(variant==0)1.5 else -1.5,n)
   weights <- if(family=="ill_conditioned_quadratic") 2^((seq_len(n)-1)%%17) else 1+(seq_len(n)-1)%%4
   target <- rep(c(2,-1),length.out=n)
   size_factors <- rep(c(4,8,16),length.out=2*n)
   counts <- size_factors
   dispersion <- if(variant==0)0.125 else 0.5
   sign <- if(variant==0)1 else -1
   value <- function(x) {
    total<-0
    if(grepl("quadratic",family,fixed=TRUE)) {
     combined<-0
     for(i in seq_len(n)) {d<-x[[i]]-target[[i]];combined<-combined+d;total<-total+0.5*weights[[i]]*d*d}
     total<-total+0.125*combined*combined
    } else if(family=="double_well") {
     for(i in seq_len(n)) {q<-x[[i]]*x[[i]]-1;total<-total+q*q}
     if(n>1) for(i in seq_len(n)) {j<-i%%n+1L;d<-x[[i]]-x[[j]];total<-total+0.125*d*d}
    } else {
     size<-1/dispersion
     for(row in seq_len(2*n)) {
      i<-(row-1L)%%n+1L;j<-i%%n+1L
      eta<-x[[i]]
      if(row>n && j!=i) eta<-eta+x[[j]]
      mu<-size_factors[[row]]*exp(eta)
      term<-(counts[[row]]+size)*log(size+mu)-size*log(size)-counts[[row]]*log(mu)
      total<-total+term
     }
    }
    sign*total
   }
   derivative <- function(x) {
    g<-numeric(n)
    if(grepl("quadratic",family,fixed=TRUE)) {
     combined<-0
     for(i in seq_len(n)) combined<-combined+(x[[i]]-target[[i]])
     for(i in seq_len(n)) g[[i]]<-weights[[i]]*(x[[i]]-target[[i]])+0.25*combined
    } else if(family=="double_well") {
     for(i in seq_len(n)) g[[i]]<-4*x[[i]]*(x[[i]]*x[[i]]-1)
     if(n>1) for(i in seq_len(n)) {j<-i%%n+1L;d<-x[[i]]-x[[j]];g[[i]]<-g[[i]]+0.25*d;g[[j]]<-g[[j]]-0.25*d}
    } else {
     size<-1/dispersion
     for(row in seq_len(2*n)) {
      i<-(row-1L)%%n+1L;j<-i%%n+1L;eta<-x[[i]]
      if(row>n && j!=i) eta<-eta+x[[j]]
      mu<-size_factors[[row]]*exp(eta)
      residual<-(size/(size+mu))*(mu-counts[[row]])
      g[[i]]<-g[[i]]+residual
      if(row>n && j!=i) g[[j]]<-g[[j]]+residual
     }
    }
    sign*g
   }
   parscale<-if(route=="mixed")rep(c(0.5,2),length.out=n) else rep(1,n)
   control<-list(maxit=100L,fnscale=sign,parscale=I(parscale),ndeps=I(rep(1e-4,n)),
                 factr=1e7,pgtol=1e-7,lmm=if(variant==0)1L else 10L)
   result<-optim(initial,value,if(gradient)derivative else NULL,method="L-BFGS-B",lower=lower,upper=upper,control=control)
   id<-paste(family,n,route,variant,gradient,sep=":")
   cases[[length(cases)+1L]]<-list(id=id,family=family,n=n,route=route,variant=variant,gradient=gradient,
      initial=I(initial),lower=I(lower),upper=I(upper),weights=I(weights),target=I(target),
      size_factors=I(size_factors),counts=I(counts),dispersion=dispersion,control=control,
      result=list(par=I(unname(result$par)),value=result$value,counts=as.list(result$counts),convergence=result$convergence,message=result$message))
  }
 }
 message(length(cases)," cases completed")
}
stopifnot(length(cases)==512L)
script_path<-sub("^--file=","",grep("^--file=",commandArgs(FALSE),value=TRUE)[[1]])
script_hash<-strsplit(system2("shasum",c("-a","256",shQuote(script_path)),stdout=TRUE),"[[:space:]]+")[[1]][[1]]
source(file.path(dirname(script_path),"oracle_json.R"))
helper_hash<-strsplit(system2("shasum",c("-a","256",shQuote(file.path(dirname(script_path),"oracle_json.R"))),stdout=TRUE),"[[:space:]]+")[[1]][[1]]
manifest<-list(serializer_sha256=helper_hash,schema=1L,identity="synthetic_v1",arithmetic_profile="r461-macos-arm-openblas-0.3.34",replacement_for="unavailable historical 2026-07-22 corpus; cases are different",
 generator="scripts/generate_stress_fixtures.R",generator_sha256=script_hash,r_version=R.version.string,platform=R.version$platform,
 blas=unname(extSoftVersion()[["BLAS"]]),threads=Sys.getenv("OPENBLAS_NUM_THREADS"),
 build_config=lapply(setNames(c("CC","CFLAGS","CXX","CXXFLAGS","FC","FFLAGS"),c("CC","CFLAGS","CXX","CXXFLAGS","FC","FFLAGS")),function(key)system2(file.path(R.home("bin"),"R"),c("CMD","config",key),stdout=TRUE)),
 session=capture.output(sessionInfo()),cases=cases)
write_verified_oracle_json(manifest,out)
cat(length(cases),"cases written to",out,"\n")

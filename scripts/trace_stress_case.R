#!/usr/bin/env Rscript
# Independent public optim trace for the ill-conditioned quadratic stress family.
args<-commandArgs(TRUE)
stopifnot(length(args)==3L,Sys.getenv("OPENBLAS_NUM_THREADS")=="1")
script<-sub("^--file=","",grep("^--file=",commandArgs(FALSE),value=TRUE)[[1]])
source(file.path(dirname(script),"oracle_json.R"))
fixture<-jsonlite::read_json(args[[1]],simplifyVector=FALSE)
case<-Filter(function(x)x$id==args[[2]],fixture$cases)
stopifnot(length(case)==1L);case<-case[[1]]
stopifnot(case$family=="ill_conditioned_quadratic")
numbers<-function(x)if(is.list(x))vapply(x,as.double,numeric(1))else as.double(x)
n<-as.integer(case$n);weights<-numbers(case$weights);target<-numbers(case$target)
control<-case$control;control$parscale<-numbers(control$parscale);control$ndeps<-numbers(control$ndeps)
if(nzchar(Sys.getenv("ORACLE_TRACE_LEVEL"))) {control$trace<-as.integer(Sys.getenv("ORACLE_TRACE_LEVEL"));control$REPORT<-1L}
value_calls<-list();gradient_calls<-list()
value<-function(x) {
 total<-0;combined<-0
 for(i in seq_len(n)){d<-x[[i]]-target[[i]];combined<-combined+d;total<-total+0.5*weights[[i]]*d*d}
 total<-(total+0.125*combined*combined)*control$fnscale
 value_calls[[length(value_calls)+1L]]<<-list(x=I(unname(x)),value=total)
 total
}
gradient<-function(x) {
 combined<-0;g<-numeric(n)
 for(i in seq_len(n))combined<-combined+(x[[i]]-target[[i]])
 for(i in seq_len(n))g[[i]]<-weights[[i]]*(x[[i]]-target[[i]])+0.25*combined
 g<-g*control$fnscale
 gradient_calls[[length(gradient_calls)+1L]]<<-list(x=I(unname(x)),gradient=I(g))
 g
}
result<-optim(numbers(case$initial),value,if(case$gradient)gradient else NULL,method="L-BFGS-B",lower=numbers(case$lower),upper=numbers(case$upper),control=control)
stopifnot(identical(writeBin(result$par,raw(),size=8),writeBin(numbers(case$result$par),raw(),size=8)),
 identical(writeBin(result$value,raw(),size=8),writeBin(case$result$value,raw(),size=8)),
 identical(as.integer(result$counts),as.integer(unlist(case$result$counts))),
 result$convergence==case$result$convergence,result$message==case$result$message)
write_verified_oracle_json(list(id=case$id,r_version=R.version.string,platform=R.version$platform,
 blas=unname(extSoftVersion()[["BLAS"]]),threads=Sys.getenv("OPENBLAS_NUM_THREADS"),session=capture.output(sessionInfo()),
 value_calls=value_calls,gradient_calls=gradient_calls,result=list(par=I(result$par),value=result$value,counts=as.list(result$counts),convergence=result$convergence,message=result$message)),args[[3]])
cat("Recorded ",length(value_calls)," value and ",length(gradient_calls)," gradient calls; complete result bits match stored R oracle\n",sep="")

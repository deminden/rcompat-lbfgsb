#!/usr/bin/env Rscript
# Public optim calls on the committed numeric GLM inputs; no external data.
script <- sub("^--file=", "", grep("^--file=", commandArgs(FALSE), value=TRUE)[[1]])
root <- normalizePath(file.path(dirname(script),".."))
source(file.path(root,"scripts","oracle_json.R"))
stopifnot(as.character(getRversion())=="4.6.2", Sys.getenv("OPENBLAS_NUM_THREADS")=="1")
sha <- function(p) strsplit(system2("shasum",c("-a","256",shQuote(p)),stdout=TRUE),"[[:space:]]+")[[1]][[1]]
metadata <- function(fixture) {
 fixture$r_version <- R.version.string; fixture$platform <- R.version$platform
 fixture$generator <- "scripts/regenerate_scientific_fixtures.R";fixture$generator_sha256 <- sha(script)
 fixture$blas <- unname(extSoftVersion()[["BLAS"]]);fixture$threads <- Sys.getenv("OPENBLAS_NUM_THREADS")
 fixture$session <- capture.output(sessionInfo())
 fixture$source <- list(description="Self-contained numeric DESeq-derived GLM inputs")
 fixture
}
row_dot <- function(x,beta) { total<-0;for(j in seq_along(beta)) total<-total+x[[j]]*beta[[j]];total }
value <- function(beta,y,x,sf,dispersion,log2=FALSE) {
 size<-1/dispersion;total<-0
 for(i in seq_along(y)) {
  eta<-row_dot(x[i,],beta);if(log2) eta<-log(2)*eta
  mu<-exp(eta+log(sf[[i]]))
  if(log2) {total<-total+(y[[i]]+size)*log(size+mu);total<-total-size*log(size);total<-total-y[[i]]*log(mu)}
  else {term<-(y[[i]]+size)*log(size+mu)-size*log(size)-y[[i]]*log(mu);total<-total+term}
 }
 total
}
gradient <- function(beta,y,x,sf,dispersion) {
 size<-1/dispersion;g<-numeric(length(beta))
 for(i in seq_along(y)) {mu<-exp(row_dot(x[i,],beta)+log(sf[[i]]));residual<-(y[[i]]+size)*mu/(size+mu)-y[[i]];for(j in seq_along(beta)) g[[j]]<-g[[j]]+x[i,j]*residual}
 g
}
run <- function(case,y,x,sf,log2=FALSE) {
 control<-lapply(case$control,unlist)
 fn<-function(beta) value(beta,y,x,sf,case$dispersion,log2)
 gr<-if(log2) NULL else function(beta) gradient(beta,y,x,sf,case$dispersion)
 result<-optim(unlist(case$initial_par),fn,gr,method="L-BFGS-B",lower=unlist(case$lower),upper=unlist(case$upper),control=control)
 case$result<-list(par=I(unname(result$par)),value=result$value,counts=as.list(result$counts),convergence=result$convergence,message=result$message)
 case
}
directory<-file.path(root,"fixtures","deseq_real_subset")
fixture<-jsonlite::read_json(file.path(directory,"optim_cases.json"),simplifyVector=FALSE)
read_table<-function(name) read.delim(file.path(directory,name),check.names=FALSE,stringsAsFactors=FALSE)
x<-as.matrix(read_table("design_matrix.tsv")[,-1]);storage.mode(x)<-"double"
sf<-read_table("size_factors.tsv")$sizeFactor;counts<-read_table("counts.tsv")
fixture$cases<-lapply(fixture$cases,function(case) run(case,as.numeric(counts[counts$gene==case$gene,-1]),x,sf))
write_verified_oracle_json(metadata(fixture),file.path(directory,"optim_cases.json"))
path<-file.path(root,"fixtures","deseq_hard_real_subset","optim_cases.json")
fixture<-jsonlite::read_json(path,simplifyVector=FALSE)
fixture$contrasts<-lapply(fixture$contrasts,function(contrast) {
 x<-do.call(rbind,lapply(contrast$design,unlist));sf<-unlist(contrast$size_factors)
 contrast$cases<-lapply(contrast$cases,function(case) run(case,unlist(case$counts),x,sf,TRUE));contrast
})
write_verified_oracle_json(metadata(fixture),path)
cat("Regenerated 68 supplied-gradient and 48 objective-only R 4.6.2 results\n")

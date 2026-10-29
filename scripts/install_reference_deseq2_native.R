#!/usr/bin/env Rscript
# Install and test public DESeq2 in the external native R 4.6.2 environment.
stopifnot(as.character(getRversion())=="4.6.2")
for(key in c("CFLAGS","CXXFLAGS","FFLAGS")) {
 flags<-system2(file.path(R.home("bin"),"R"),c("CMD","config",key),stdout=TRUE)
 stopifnot(grepl("-mcpu=native",flags,fixed=TRUE))
}
if(!requireNamespace("BiocManager",quietly=TRUE)) install.packages("BiocManager",repos="https://cloud.r-project.org")
BiocManager::install("DESeq2",version="3.23",ask=FALSE,update=FALSE,type="source",force=TRUE,Ncpus=6)
suppressPackageStartupMessages(library(DESeq2))
stopifnot(grepl("R 4.6.2",packageDescription("DESeq2")$Built,fixed=TRUE))
counts<-matrix(as.integer(rep(seq(10,85,by=5),6)*rep(c(1,1.1,0.9,2,2.1,1.9),each=16)),nrow=16)
rownames(counts)<-paste0("gene",seq_len(16));colnames(counts)<-paste0("sample",seq_len(6))
metadata<-S4Vectors::DataFrame(condition=factor(rep(c("a","b"),each=3)),row.names=colnames(counts))
dds<-DESeqDataSetFromMatrix(countData=counts,colData=metadata,design=~condition)
sizeFactors(dds)<-rep(1,6);dispersions(dds)<-rep(0.2,16)
.reference_optim_calls<-0L
trace("optim",where=asNamespace("stats"),tracer=quote(.reference_optim_calls<<-.reference_optim_calls+1L),print=FALSE)
fitted<-nbinomWaldTest(dds,maxit=1,useOptim=TRUE,quiet=TRUE)
untrace("optim",where=asNamespace("stats"))
stopifnot(all(is.finite(coef(fitted))),.reference_optim_calls>0)
cat(R.version.string,"\nDESeq2 ",as.character(packageVersion("DESeq2")),"\n",sep="")
cat("Native reference smoke fit: 16 genes; ",.reference_optim_calls," optimizer fallback calls; all coefficients finite\n",sep="")
print(sessionInfo())

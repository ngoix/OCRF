###############
#  LIBRARIES  #
###############

library(MASS)
library(cluster)
library(survival)
library(randomForest)
library(Hmisc)

#########################################################################################
#########################################################################################
if (exists("Rand") ) rm(Rand)
Rand <- function(tab,adjust=T) {

##########################################################################
# The function computes the (adjusted) Rand index between two partitions #
# Copyright Steve Horvath and Luohua Jiang, UCLA, 2003                   #
##########################################################################

    # helper function
    choosenew <- function(n,k) {
        n <- c(n); out1 <- rep(0,length(n));
        for (i in c(1:length(n)) ){
            if ( n[i]<k ) {out1[i] <- 0}
            else {out1[i] <- choose(n[i],k) }
        }
        out1
    }

    a <- 0; b <- 0; c <- 0; d <- 0; nn <- 0
    n <- nrow(tab)
    for (i in 1:n) {
        for(j in 1:n) {
            a <- a+choosenew(tab[i,j],2)
            nj <- sum(tab[,j])
            c <- c+choosenew(nj,2)
        }
        ni <- sum(tab[i,])
        b <- b+choosenew(ni,2)
        nn <- nn+ni
    }
    if(adjust==T) {
        d <- choosenew(nn,2)
        adrand <- (a-(b*c/n)/d)/(0.5*(b+c/n)-(b*c/n)/d)
        adrand
    } else {
        b <- b-a
        c <- c/n-a
        d <- choosenew(nn,2)-a-b-c
        rand <- (a+d)/(a+b+c+d)
        rand
    }
}

#########################################################################################
#########################################################################################
if (exists("pamNew") ) rm(pamNew)
pamNew <- function (x, k, diss1 = inherits(x, "dist"), metric1 = "euclidean")
{

#############################################################################################################
# A new pam clustering function which corrects the clustering membership based on the sillhouette strength. #
# The clustering membership of an observation with a negative sillhouette strength is reassigned to its     #
# neighboring cluster.                                                                                      #
# The inputs of the function are similar to the original 'pam' function.                                    #
# The function returns a vector of clustering labels.                                                       #
# Copyright 2003 Tao Shi and Steve Horvath (last modified 10/31/03)                                         #
#############################################################################################################

    if (diss1)
    {
        if (!is.null(attr(x, "Labels"))) { original.row.names <- attr(x, "Labels")}
        names(x) <- as.character(c(1:attr(x, "Size")))
    }
    else
    {
        if(!is.null(dimnames(x)[[1]])) { original.row.names <- dimnames(x)[[1]]}
        row.names(x) <- as.character(c(1:dim(x)[[1]]))
    }
    pam1 <- pam(x,k,diss=diss1, metric=metric1)
    label2 <- pam1$clustering
    silinfo1 <- pam1$silinfo$widths
    index1 <- as.numeric(as.character(row.names(silinfo1)))
    silinfo2 <- silinfo1[order(index1),]
    labelnew <- ifelse(silinfo2[,3]<0, silinfo2[,2], silinfo2[,1])
    names(labelnew) <- original.row.names
    labelnew
}


###############################################################################################
###############################################################################################
if (exists("collect.garbage") ) rm(collect.garbage)
collect.garbage <- function(){
## The following function collects garbage until the memory is clean.
## Usage: 1. immediately call this function after you call a function or
##        2. rm()
	while (gc()[2,4] != gc()[2,4]){}
}


###############################################################################################
###############################################################################################

if (exists("RFdist") ) rm(RFdist)
RFdist <- function(datRF, mtry1, no.tree, no.rep, addcl1=T, addcl2=T, imp=T, oob.prox1=T, proxConver=F) {

####################################################################
# Unsupervised randomForest function                               #
# Return a list "distRF" containing some of the following 6 fields #
#  depending on the options specified:                             #
#  (1) cl1:  addcl1 distance (sqrt(1-RF.proxAddcl1))               #
#  (2) err1: error rate                                            #
#  (3) imp1: variable importance for addcl1                        #
#  (4) prox1Conver: a matrix containing two convergence meausres   #
#                   for addcl1 proximity                           #
#                   a). max( abs( c(aveprox(N))-c(aveprox(N-1))))  #
#                   b). mean((c(aveprox(N))-c(aveprox(N-1)))^2)    #
#                   where N is number of forests (no.rep).         #
#  (5) cl2, (6) err2, (7)imp2 and (8) prox2Conver for addcl2       #
# Copyright Steve Horvath and Tao Shi (2004)                       #
####################################################################


    synthetic1 <- function(dat) {
        sample1 <- function(X)   { sample(X, replace=T) }
        g1      <- function(dat) { apply(dat,2,sample1) }
        nrow1 <- dim(dat)[[1]];
        yy <- rep(c(1,2),c(nrow1,nrow1) );
        data.frame(cbind(yy,rbind(dat,data.frame(g1(dat)))))
    }

    synthetic2 <- function(dat) {
        sample2 <- function(X)   { runif(length(X), min=min(X), max =max(X)) }
        g2      <- function(dat) { apply(dat,2,sample2) }
        nrow1 <- dim(dat)[[1]];
        yy <- rep(c(1,2),c(nrow1,nrow1) );
        data.frame(cbind(yy,rbind(dat,data.frame(g2(dat)))))
    }

    cleandist <- function(x) {
        x1 <- as.dist(x)
        x1[x1<=0] <- 0.0000000001
        as.matrix(x1)
    }

    nrow1 <- dim(datRF)[[1]]
    ncol1 <- dim(datRF)[[2]]
    RFproxAddcl1 <- matrix(0,nrow=nrow1,ncol=nrow1)
    RFproxAddcl2 <- matrix(0,nrow=nrow1,ncol=nrow1)
    RFprox1Conver <- cbind(1:no.rep,matrix(0,(no.rep),3))
    RFprox2Conver <- cbind(1:no.rep,matrix(0,(no.rep),3))
    RFimportance1 <- matrix(0, nrow=ncol1, ncol=4)
    RFimportance2 <- matrix(0, nrow=ncol1, ncol=4)
    RFerrrate1 <- 0
    RFerrrate2 <- 0
    rep1 <- rep(666,2*nrow1)

    #addcl1
    if (addcl1) {
        for (i in c(0:no.rep)) {
            index1 <- sample(c(1:(2*nrow1)))
            rep1[index1] <-  c(1:(2*nrow1))
            datRFsyn <- synthetic1(datRF)[index1,]
            yy <- datRFsyn[,1]
            RF1 <- randomForest(factor(yy)~.,data=datRFsyn[,-1], ntree=no.tree, oob.prox=oob.prox1, proximity=TRUE,do.trace=F,mtry=mtry1,importance=imp)
            collect.garbage()
		RF1prox <- RF1$proximity[rep1,rep1]
            if (i > 0) {
                if (i > 1){
                    xx <- ((RFproxAddcl1 + (RF1prox[c(1:nrow1),c(1:nrow1)]))/i) - (RFproxAddcl1/(i-1))
                    yy <- mean( c(as.dist((RFproxAddcl1 + (RF1prox[c(1:nrow1),c(1:nrow1)]))/i)))
                    RFprox1Conver[i,2] <- max(abs(c(as.dist(xx))))
                    RFprox1Conver[i,3] <- mean((c(as.dist(xx)))^2)
                    RFprox1Conver[i,4] <- yy
                }
                RFproxAddcl1 <- RFproxAddcl1 + (RF1prox[c(1:nrow1),c(1:nrow1)])
                if(imp) { RFimportance1 <- RFimportance1+ 1/no.rep*(RF1$importance) }
                RFerrrate1 <- RFerrrate1+ 1/no.rep*(RF1$err.rate[no.tree])
            }
        }
    }

    # addcl2
    if (addcl2) {
        for (i in c(0:no.rep)) {
            index1 <- sample(c(1:(2*nrow1)))
            rep1[index1] <-  c(1:(2*nrow1))
            datRFsyn <- synthetic2(datRF)[index1,]
            yy <- datRFsyn[,1]
            RF2 <- randomForest(factor(yy)~.,data=datRFsyn[,-1], ntree=no.tree, oob.prox=oob.prox1, proximity=TRUE,do.trace=F,mtry=mtry1,importance=imp)
            collect.garbage()
		RF2prox <- RF2$proximity[rep1,rep1]
            if (i > 0) {
                if (i > 1){
                    xx <- ((RFproxAddcl2 + (RF2prox[c(1:nrow1),c(1:nrow1)]))/i) - (RFproxAddcl2/(i-1))
                    yy <- mean( c(as.dist((RFproxAddcl2 + (RF2prox[c(1:nrow1),c(1:nrow1)]))/i)))
                    RFprox2Conver[i,2] <- max(abs(c(as.dist(xx))))
                    RFprox2Conver[i,3] <- mean((c(as.dist(xx)))^2)
                    RFprox2Conver[i,4] <- yy
                }
                RFproxAddcl2 <- RFproxAddcl2 + (RF2prox[c(1:nrow1),c(1:nrow1)])
                if(imp) { RFimportance2 <- RFimportance2+ 1/no.rep*(RF2$importance)}
                RFerrrate2 <- RFerrrate2+ 1/no.rep*(RF2$err.rate[no.tree])
            }
        }
    }

    distRFAddcl1 <- cleandist(sqrt(1-RFproxAddcl1/no.rep))
    distRFAddcl2 <- cleandist(sqrt(1-RFproxAddcl2/no.rep))

    distRF <- list(cl1=NULL, err1=NULL, imp1=NULL, prox1Conver=NULL,
                   cl2=NULL, err2=NULL, imp2=NULL, prox2Conver=NULL)
    if(addcl1) {
        distRF$cl1 <- distRFAddcl1
        distRF$err1 <- RFerrrate1
        if(imp) distRF$imp1 <- RFimportance1
        if(proxConver) distRF$prox1Conver <- RFprox1Conver
    }
    if(addcl2) {
        distRF$cl2 <- distRFAddcl2
        distRF$err2 <- RFerrrate2
        if(imp) distRF$imp2 <- RFimportance2
        if(proxConver) distRF$prox2Conver <- RFprox2Conver
    }
    distRF
}

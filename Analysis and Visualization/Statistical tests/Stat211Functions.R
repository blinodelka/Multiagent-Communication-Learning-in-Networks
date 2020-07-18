##
#####	Collection of functions written by D. Gillen that are used
#####	throughout Stat 211
#####
#####	Author: D. Gillen
#####	Course: Stat 211, Winter 2018
##
##
#####	Helper function to return vector for binary test
##
ifelse1 <- function (test, yes, no){
    if (test) yes
    else no
}

##
#####  Function to produce CIs for LM parameters (or exponentiated parameters)
##
lmCI <- function( model, expcoef=FALSE, robust=FALSE ){
	coef <- summary( model )$coef[,1]
	se <- ifelse1( robust, robust.se.lm(model)[,2], summary( model )$coef[,2] )
	tvalue <- coef / se
	pvalue <- 2*(1-pt(abs(tvalue), model$df.residual))
	if( expcoef ){
		ci95.lo <- exp( coef - qt(.975, model$df.residual) * se )
		ci95.hi <- exp( coef + qt(.975, model$df.residual) * se )
		est <- exp( coef )
	}
	else{
		ci95.lo <- coef - qt(.975, model$df.residual) * se
		ci95.hi <- coef + qt(.975, model$df.residual) * se
		est <- coef
	}
	rslt <- round( cbind( est, ci95.lo, ci95.hi, tvalue, pvalue ), 4 )
	colnames( rslt ) <- ifelse1( 	robust, 	
					c("Est", "robust ci95.lo", "robust ci95.hi", "robust t value", "robust Pr(>|t|)"),
					c("Est", "ci95.lo", "ci95.hi", "t value", "Pr(>|t|)") )			
	colnames( rslt )[1] <- ifelse( expcoef, "exp( Est )", "Est" )
	rslt
	}


##
#####	Function to plot dfBetas resulting from a lm() object
##
plot.dfbeta <- function( dfbeta.fit, labels ){
	oldmar <- par()$mar
	par( mar=c(5, 4, 3+.75*dim(dfbeta.fit)[2], 2) + 0.1 )
	plot( c(1,dim(dfbeta.fit)[1]*1.1), range(dfbeta.fit)*1.1, xlab="Obersvation", ylab="dfBeta", type="n" )
	for( i in 2:dim(dfbeta.fit)[2] ){
			points( 1:dim(dfbeta.fit)[1], dfbeta.fit[,i], col=i )
			text( 1:dim(dfbeta.fit)[1]+1, dfbeta.fit[,i]+.1, labels=labels, col=i )
			mtext( colnames( dfbeta.fit )[i], col=i, line=-1+i )
	}  
	abline( h=c(-1,1)*(2/sqrt(dim(dfbeta.fit)[1])), col="red", lwd=2 )
	par( mar=oldmar )
	}
	

##
##### 	Compute robust (sandwich) variance-covariance estimate for a LM
##
robust.vcov.lm <- function( lm.obj ){
  X <- model.matrix( lm.obj )
  eps <- lm.obj$residuals
  robust.cov <- solve( t(X)%*%X ) %*%( t(X) %*% diag(eps^2) %*% X ) %*% solve( t(X)%*%X )
  dimnames( robust.cov ) <- dimnames( vcov(lm.obj) )
  return( robust.cov )
}

##
#####
#####	robust.se.lm() is a function to compute the Huber-White sandwich variance estimator
#####	for the linear regression model
#####	
##
robust.se.lm <- function( model) { 
  s <- summary( model) 
  X <- model.matrix( model )
  sandwich.cov <- robust.vcov.lm( model )
  sand.se <- sqrt( diag( sandwich.cov )) 
  t <- model$coefficients/sand.se
  p <- 2*pt( -abs( t ), dim(X)[1]-dim(X)[2] ) 
  ci95.lo <- model$coefficients - qt( .975, dim(X)[1]-dim(X)[2] ) * sand.se
  ci95.hi <- model$coefficients + qt( .975, dim(X)[1]-dim(X)[2] ) * sand.se
  rslt <- cbind( model$coefficients, sand.se, ci95.lo, ci95.hi, t, p ) 
  dimnames(rslt)[[2]] <- c( dimnames( s$coefficients )[[2]][1], "Robust SE", "ci95.lo", "ci95.hi", dimnames( s$coefficients )[[2]][3:4] ) 
  rslt 
} 


##
#####  Function to exponentiate coefficients and produces CIs for GLMs
##
glmCI <- function( model, transform=TRUE, robust=FALSE ){
	link <- model$family$link
	coef <- summary( model )$coef[,1]
	se <- ifelse1( robust, robust.se.glm(model)[,2], summary( model )$coef[,2] )
	zvalue <- coef / se
	pvalue <- 2*(1-pnorm(abs(zvalue)))

	if( transform & is.element(link, c("logit","log")) ){
		ci95.lo <- exp( coef - qnorm(.975) * se )
		ci95.hi <- exp( coef + qnorm(.975) * se )
		est <- exp( coef )
	}
	else{
		ci95.lo <- coef - qnorm(.975) * se
		ci95.hi <- coef + qnorm(.975) * se
		est <- coef
	}
	rslt <- round( cbind( est, ci95.lo, ci95.hi, zvalue, pvalue ), 4 )
	colnames( rslt ) <- ifelse1( 	robust, 	
					c("Est", "robust ci95.lo", "robust ci95.hi", "robust z value", "robust Pr(>|z|)"),
					c("Est", "ci95.lo", "ci95.hi", "z value", "Pr(>|z|)") )			
	colnames( rslt )[1] <- ifelse( transform & is.element(link, c("logit","log")), "exp( Est )", "Est" )
	rslt
	}


##
#####  Function to collapse grouped binary data to binomial
##
collapse <- function( data, outcome ){
	index <- (1:length(names(data)))[ names(data)==outcome ]
	y <- data[,index]
	covnames <- names( data )[-index]
	data <- data[,-index]
	if( is.null( dim( data ) ) ){
		rslt <- aggregate( y, list(data), FUN=length)
		rslt <- as.data.frame( cbind( rslt, aggregate( y, list(data), FUN=sum)[dim(rslt)[2]] ) )
	}
	else{
		rslt <- aggregate( y, data, FUN=length)
		rslt <- as.data.frame( cbind( rslt, aggregate( y, data, FUN=sum)[dim(rslt)[2]] ) )
	}
	names( rslt ) <- c( covnames, "n", paste("n.", outcome, sep="") )
	rslt
}
##
#####	Function to compute deviance (LR) test p-Value
##
lrtest <- function( fit1, fit2 ){
	cat( "\nAssumption: Model 1 nested within Model 2\n\n" )
	rslt <- anova( fit1, fit2 )
	rslt <- cbind( rslt, c("", round( pchisq( rslt[2,4], rslt[2,3], lower.tail=FALSE ), 4 ) ) )
	rslt[,2] <- round( rslt[,2], 3 )
	rslt[,4] <- round( rslt[,4], 3 )
	rslt[1,3:4] <- c( "", "" )
	names( rslt )[5] <- "pValue"
	rslt
}
##
#####	H-L goodness of fit test
##
binary.gof <- function( fit, ngrp=10, print.table=TRUE ){
	y <- fit$y
	phat <- fitted( fit )
	fittedgrps <- cut( phat, quantile( phat, seq(0,1,by=1/ngrp) ), include.lowest=TRUE )
	n <- aggregate( y, list( fittedgrps ), FUN=length )[,2]
	Obs <- aggregate( y, list( fittedgrps ), FUN=sum )[,2]
	Exp <- aggregate( phat, list( fittedgrps ), FUN=sum )[,2]
	if( print.table==TRUE ){
		cat( "\nFitted Probability Table:\n\n" )
		rslt <- as.data.frame( cbind( 1:ngrp, n, Obs, Exp ) )
		names( rslt )[1] <- "group"
		print( rslt )
	}
	chisqstat <- sum( (Obs - Exp)^2 / ( Exp*(1-Exp/n) ) )
	df <- ngrp - 2
	pVal <- pchisq( chisqstat, df, lower.tail=FALSE )
	cat( "\n Hosmer-Lemeshow GOF Test:\n\n" )
	cbind( chisqstat, df, pVal )
}

##
##### 	Compute robust (sandwich) variance-covariance estimate for a GLM
##
robust.vcov.glm <- function(glm.obj){
	if (is.matrix(glm.obj$x)) 
		xmat<-glm.obj$x
	else {
		mf<-model.frame(glm.obj)
		xmat<-model.matrix(terms(glm.obj),mf)		
	}
	umat <- residuals(glm.obj,"working")*glm.obj$weights*xmat
	modelv<-summary(glm.obj)$cov.unscaled
	robust.cov <- modelv%*%(t(umat)%*%umat)%*%modelv
	dimnames( robust.cov ) <- dimnames( vcov(glm.obj) )
	return( robust.cov )
}
	
	
##
#####	Function to compute robust se for glms
##
robust.se.glm <- function(glm.obj){
	## 	Compute robust (sandwich) variance estimate
	robust.cov <- robust.vcov.glm(glm.obj)
	
	##	Format the model output with p-values and CIs
	s <- summary( glm.obj) 
	robust.se <- sqrt( diag( robust.cov )) 
	z <- glm.obj$coefficients/robust.se
	p <- 2*pnorm( -abs( z ) ) 
	ci95.lo <- glm.obj$coefficients - qnorm( .975 ) * robust.se
	ci95.hi <- glm.obj$coefficients + qnorm( .975 ) * robust.se
	rslt <- cbind( glm.obj$coefficients, robust.se, ci95.lo, ci95.hi, z, p ) 
	dimnames(rslt)[[2]] <- c( dimnames( s$coefficients )[[2]][1], "Robust SE", "ci95.lo", "ci95.hi", "z value", "Pr(>|z|)" ) 
	rslt 
	}

##
#####	Function to summarize the multinomial fit
##
summ.mfit <- function( model ){
	s <- summary( model )
	for( i in 1:length(model$coef) ){
		cat( "\nLevel ", model$lev[i+1],  "vs. Level ", model$lev[1], "\n" )
		coef <- s$coefficients[i,]
		rrr <- exp( coef )
		se <- s$standard.errors[i,]
		zStat <- coef / se
		pVal <- 2*pnorm( abs(zStat), lower.tail=FALSE )
		ci95.lo <- exp( coef - qnorm(.975)*se )
		ci95.hi <- exp( coef + qnorm(.975)*se )
		rslt <- cbind( rrr, se, zStat, pVal, ci95.lo, ci95.hi )
		print( round( rslt, 3 ) )
	}
}

#
##
#####	Function to summarize a proportional odds fit
##
olrCI <- function( model, transform=TRUE ){
		coef <- summary( model )$coef[ 1:length(model$coef), ]
		if( is.null( dim( coef ) ) ) coef <- t( as.matrix(coef) )
		zStat <- as.matrix(coef[,3])
		pVal <- 2*pnorm( abs(zStat),lower.tail=FALSE )
		if( transform ){
			ci95.lo <- exp( coef[,1] - qnorm(.975) * coef[,2] )
			ci95.hi <- exp( coef[,1] + qnorm(.975) * coef[,2] )
			est <- exp( coef[,1] )
		}
		else{
			ci95.lo <- coef[,1] - qnorm(.975) * coef[,2]
			ci95.hi <- coef[,1] + qnorm(.975) * coef[,2]
			est <- coef[,1]
		}
		rslt <- round( cbind( est, ci95.lo, ci95.hi, zStat, pVal), 4 )
		if( transform ) colnames( rslt ) <- c( "exp( Est )", "ci95.lo", "ci95.hi", "z value", "Pr(>|z|)" )
			else colnames( rslt ) <- c( "est", "ci95.lo", "ci95.hi", "z value", "Pr(>|z|)" )
		rslt
		}
		


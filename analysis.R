library(data.table)
library(multcomp)

#######################################################
# Analysis for the accuracies
#setwd('data')
x <- fread('accuracies.csv')

# original
x.mean <- colMeans(x) 
x.cov <- cov(x) 
x.names <- names(copy(x))

# ascending mean
y.mean <- x.mean[order(x.mean)]
y.cov <- x.cov[order(x.mean),order(x.mean)]
y.names <- x.names[order(x.mean)]

# precontrasts - i.e., vectors to construct pairwise contrasts
y.precontr <- rep(1,8); names(y.precontr) <- y.names

# all values contrasted with nbsvm
point.mod <- glht(parm(y.mean, y.cov),
                  linfct=diag(8))
point.cis <- confint(point.mod)
diff.mod <- glht(parm(y.mean, y.cov),
                 linfct=contrMat(y.precontr,
                                 type="Dunnett",
                                 base=8))
diff.cis <- confint(diff.mod)
write.csv(point.cis$confint, 'acc-cis.csv')
write.csv(diff.cis$confint, 'acc-diff-cis.csv')

#######################################################
# Analysis for the discordances
count.df <- fread('counts.csv')
x.mean <- (count.df$pos.calls - count.df$true.pos) / count.df$pop
names(x.mean) <- count.df$model

# original
x.cov <- cov(x) 
x.names <- names(x.mean)

# descending mean (by abs)
ord <- order(abs(x.mean), decreasing=T)
y.mean <- x.mean[ord]
y.cov <- x.cov[ord, ord]
y.names <- x.names[ord]

# precontrasts - i.e., vectors to construct pairwise contrasts
y.precontr <- rep(1,8); names(y.precontr) <- y.names

# getting the CIs for the point estimates
point.mod <- glht(parm(y.mean, y.cov),
                  linfct=diag(8))
point.cis <- as.data.frame(confint(point.mod)$confint) * 100
point.cis$model <- y.names
write.csv(point.cis, 'disc-cis.csv')

# getting the CIs for the differences in point estimates
diff.mod <- glht(parm(y.mean, y.cov),
                 linfct=contrMat(y.precontr,
                                 type="Dunnett",
                                 base=8))
diff.cis <- confint(diff.mod)
write.csv(diff.cis$confint * 100, 'disc-diff-cis.csv')

# setwd("./Repositories/ctsmr-tmb-experiments")
# setwd("~/.")
print(getwd())
library(ctsmr)
library(splines)
library(tidyverse)
library(ggfortify)
library("grid")

# My learnins with CTSMR:
# done't restrict variables too much, but sometimes it is also good to do
# start guesses are really important for convergence.
# Every time we add an input, it takes way more time unless we have good start guesses
# There is great multiplicity in solutions. E.g. it will gladly increase/change C and R significantly,
#   if the splines (given good initial guesses for coefficients) can provide a better fit
# Also, it really helps to smoothen inputs to the model as well.

## read data
X <- read.csv("data/sde.csv", sep = ",", header = TRUE)
X <- subset(X, select = c(t, Rt, Pt, Tc, OD, Ta, day_filter, night_filter, defrost_filter))
names(X) <- c("t", "Rt", "Pt", "yTc", "OD", "Ta", "day", "night", "defrost")
setpoint <- -18.5

model <- ctsm()
model$addSystem(dTfood ~ (1 / C_f * (Tc - Tfood)) * dt + exp(e1) * dw1)
model$addSystem(dTc ~ (1 / (R_cf * C_c) * (Tfood - Tc) + 1 / ((Rt * (day * a1 + night * a2)) * C_c) * (Ta - Tc) - (eta / C_c) * Pt * OD + 1 / C_c * defrost * epsilon) * dt + exp(e2) * dw2)

model$addInput(Rt, Pt, Ta, day, night, defrost, OD)
model$addObs(yTc ~ Tc)
model$setVariance(yTc ~ exp(e3))

model$setParameter(Tc = c(init = setpoint))
model$setParameter(Tfood = c(init = setpoint))

model$setParameter(C_f = c(init = 5.5, lb = 5, ub = 10))
model$setParameter(C_c = c(init = 0.2, lb = 0.0001, ub = 0.5))
# model$setParameter(C_c = c(init = 0.129))
model$setParameter(R_cf = c(init = 5.1, lb = 5, ub = 10))
# model$setParameter(R_cf = c(init = 4.910))
model$setParameter(eta = c(init = 3, lb = 0.1, ub = 10))
# model$setParameter(eta = c(init = 2.383))
model$setParameter(epsilon = c(init = 6, lb = 1, ub = 50))

model$setParameter(a1 = c(init = 1.2, lb = 0.5, ub = 2))
# model$setParameter(a1 = c(init = 1.246))
model$setParameter(a2 = c(init = 1.2, lb = 0.5, ub = 2))
# model$setParameter(a2 = c(init = 1.5))

model$setParameter(e1 = c(init = 0, lb = -50, ub = 2))
model$setParameter(e2 = c(init = 0, lb = -50, ub = 2))
model$setParameter(e3 = c(init = 0, lb = -50, ub = 2))


# model$options$maxNumberOfEval = 1000
# model$options$nIEKF = 7
# model$options$iEKFeps = 1E-05
# model$options$odeeps = 1E-05
fit <- model$estimate(X, firstorder = TRUE, threads = 4)
summary(fit, extended = TRUE)
print(round(fit$xm, 3))


## Calculate the one-step predictions of the state (i.e. the residuals)
tmp <- predict(fit, n.ahead = 1)[[1]]
## Calculate the residuals and put them with the data in a data.frame X
X$residuals <- X$yTc - tmp$output$pred$yTc
X$TcHat <- tmp$output$pred$yTc
X$TfoodHat <- tmp$state$pred$Tfood

acf(X$residuals, lag.max = 6 * 12, main = "Residuals ACF")
## The cumulated periodogram
cpgram(X$residuals, main = "Cumulated periodogram")

## Plot the auto-correlation function and cumulated periodogram in a new window
par(mfrow = c(1, 2), mar = c(3, 3.5, 0.1, 0.1))
require(gtable)
require(ggplot2)
## The blue lines indicates the 95 confidence interval, meaning that if it is
## white noise, then approximately 1 out of 20 lag correlations will be slightly outside
# tiff("tex/figures/test.tiff", units = "in", width = 5, height = 5, res = 300)
png(filename = "tex/figures/2ndFreezerModelValidation.png", units = "in", width = 5, height = 2.6, res = 300)
bacf <- acf(X$residuals, lag.max = 6 * 12, plot = FALSE)
bacfdf <- with(bacf, data.frame(lag, acf))
q <- ggplot(data = bacfdf, mapping = aes(x = lag, y = acf)) +
  geom_hline(aes(yintercept = 0)) +
  geom_segment(mapping = aes(xend = lag, yend = 0)) +
  theme(text = element_text(size = 15))
g1 <- ggplotGrob(q)
cp <- ggcpgram(X$residuals) + theme(text = element_text(size = 15))
g2 <- ggplotGrob(cp)
# plot(X$t, X$residuals, xlab="residuals", ylab="", type="n")
# lines(X$t, X$residuals)
g <- gtable:::cbind_gtable(g1, g2, "min")
panels <- g$layout$t[grep("panel", g$layout$name)]
g$heights[panels] <- unit(c(5, 5), "cm")
grid.newpage()
grid.draw(g)
# dev.copy(png, filename = "tex/figures/2ndFreezerModelValidation.png")
dev.off()
##
par(mfrow = c(1, 1))
plot(X$t, X$yTc, xlab = "Time.", ylab = "Temperature", type = "n") # , ylim=c(-20,20))
lines(X$t, X$yTc)
lines(X$t, X$TcHat, col = 2)
# lines(X$t, X$bs4, col=3)
legend("bottomright", c("Measured", "Predicted"), col = 1:2, lty = 1)

# ##
# par(mfrow = c(1, 1))
# plot(X$t, X$yTc, xlab = "Time.", ylab = "", type = "n", ylim = c(-25, 0)) # , xlim=c(0,100))
# lines(X$t, X$TfoodHat)
# lines(X$t, X$yTc, col = 2)
# lines(X$t, X$TcHat, col = 3)
# legend("bottomright", c("Tfood", "yTc", "yTcHat"), col = 1:3, lty = 1)

## simulate model
c_f <- fit$xm["C_f"]
c_c <- fit$xm["C_c"]
r_cf <- fit$xm["R_cf"]
eta <- fit$xm["eta"]
epsilon <- fit$xm["epsilon"]
a1 <- fit$xm["a1"]
a2 <- fit$xm["a2"]
day <- X$day
night <- X$night

Rt <- X$Rt * (day * a1 + night * a2)
od <- X$OD
Pt <- X$Pt
Ta <- X$Ta
defrost <- X$defrost
dt <- 0.25

Tf <- c(dim(X)[1])
Tc <- c(dim(X)[1])
Tf[1] <- setpoint
Tc[1] <- setpoint

for (i in 2:dim(X)[1]) {
  # Tf[i] <- Tf[i - 1] + dt * 1 / (c_f * r_cf) * (Tc[i - 1] - Tf[i - 1])
  Tf[i] <- Tf[i - 1] + dt * 1 / c_f * (Tc[i - 1] - Tf[i - 1])
  Tc[i] <- Tc[i - 1] + dt * 1 / c_c * (1 / r_cf * (Tf[i - 1] - Tc[i - 1]) + 1 / Rt[i] * (Ta[i] - Tc[i - 1]) - eta * od[i] * Pt[i] + epsilon * defrost[i])
}

par(mfrow = c(1, 1))
plot(X$t, Tf, xlab = "Time.", ylab = "Celsius", type = "n", ylim = c(-25, 0))
lines(X$t, Tf)
lines(X$t, Tc, col = 2)
lines(X$t, X$Tc, col = 3)
legend("bottomright", c("Tf-sim", "yTc-sim", "yTc-meas"), col = 1:3, lty = 1)

print(round(fit$xm, 3))

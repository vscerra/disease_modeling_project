##===========================================================
## independent_SIS.R
## Author: Veronica Scerra
## Last Updated: 2025-11-14
##===========================================================
## SIS (Susceptible-Infected-Susceptible) Model - no immunity

## Modeling a curable disease in a closed population where
## recovery does not induce immunity (e.g. a bacterial STI) and
## infection status does not change behvaior:
## - S: Susceptible individuals
## - I: Infected (and infectous) individuals
## - S: Post-infection, individuals are once again susceptible,
##    no immunity is conferred by infection
## - Node heterogeneity is represented by two-level risk group designation
## with assortative mixing by level
## - 'One-mode' network - ties are allowed between all node types

## This model assumes:
## - No immunity
## - No influence of status on behavior
##===========================================================
library("EpiModel")
##help(package = "EpiModel")

# set seed
set.seed(12345)

##=======================
## Initialize Network
##=======================
## set up network with no ties with 1000 persons in two equal-sized risk groups
## risk group is a binary variable used to control infection risk
##      as a function of network structurE
nw <- network:: network.initialize(n = 1000, directed = FALSE)
nw <- network:: set.vertex.attribute(nw, "risk", rep(0:1, each = 500))

##==========================================
## Network model estimation and diagnostics
##==========================================
## ERGM formation
formation <- ~ edges + nodefactor("risk") + nodematch("risk") + concurrent

## target statistics ()
target.stats <- c(250, 375, 225, 100)

## ERGM dissolution coeffs
coef.diss <- dissolution_coefs(dissolution = ~ offset(edges), duration = 80)
coef.diss

## Estimate the model - inputs are base network, formulation formula,
##   formation target statistics, and dissolution coefficient
est1 <- netest(nw, formation, target.stats, coef.diss)

##==========================
## Diagnosing network fits
##==========================

dx <- netdx(est1, nsims = 10, nsteps = 1000)

## Edge duration shows mean age of partnerships across simulations.
## Edge dissolution at each time-step, should be the inverse of
## expected duration (80 timestep edge should lead to a dissolution
## rate of 1/80 at each ts)

## plot dx to see time series of target statistics against targets
## Black lines are targets, solid lines are mean, shading indicates IQR
plot(dx)

par(mfrow = c(1, 2))
plot(dx, type = "duration")
plot(dx, type = "dissolution")

##=======================================
## Epidemic model setup and simulation
##=======================================
## for initial conditions, use i.num argument to set up the
## number of infected at the start
## for a prevalence of 5% equally distributed over risk groups:
init <- init.net(i.num = 50)

## basic SIS model requires 3 parameters:
## - infection probability (inf.prob)
## - act rate (act.rate) the mean freq of acts per person per time step
##       is the mean # of partnerships per person multiplied by the
##       act rate parameter
## - recovery rate (rec.rate) prob that infected person recovers in a give ts 
param <- param.net(inf.prob = 0.1, act.rate = 5, rec.rate = 0.02)

## control settings contain structural features of the model
control <- control.net(type = "SIS", nsteps = 500, nsims = 10, epi.by = "risk")

## pass the fitted network model object
sim1 <- netsim(est1, param, init, control)
sim1

## Show summary (mean/std) of sims at ts=500
summary(sim1, at = 500)

## plot prevalence of the compartments in the model across sims (means, iqr)
plot(sim1)

## plot incidence of infection and recovery
plot(sim1, y = c("si.flow", "is.flow"), legend = TRUE)

## to compare results by risk group pass stratefied outputs to y
plot(sim1, y = c("i.num.risk0", "i.num.risk1"), legend = TRUE)

## static network plots at different time steps can show the patterns of
## partnership formation and infection spread over those partnerships.
## The 'col' argument handles the color coding of I (red) and S (blue)
par(mfrow = c(1, 2), mar = c(0, 0, 1, 0))
plot(sim1, type = "network", at = 1, sims = "mean", col.status = TRUE, main = "Prevalence at t1")
plot(sim1, type = "network", at = 500, sims = "mean", col.status = TRUE, main = "Prevalence at t500")
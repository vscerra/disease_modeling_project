##===========================================================
## dependent_SI.R
## Author: Veronica Scerra
## Last Updated: 2025-11-15
##===========================================================
## SI (Susceptible-Infected) Model

## Modeling a curable disease in an open population where
## recovery does not induce immunity (e.g. a bacterial STI)
## Dependencies include:
## - Demographic dynamics: births and deaths, entry of new unpartnered persons
##      - Replenishes S and removes I
## - Infection-adaptive behavior: I status can also influence the probability
##      of partnerhips and behavior within partnerships
##      - Influences both the network model and epidemic process
## - Infection-dependent vital dynamics: Infection status can affect fertility
##      and mortality
##      - allowing the epidemic to feed back on to demographic processes

## Changes in any of the vital dynamics require the network to be updated 
##      before simulating each step

##===========================================================
library("EpiModel")
##help(package = "EpiModel")

##=========================
## Network Initialization
##=========================
## Simulating a bipartite network where all ties must be BETWEEN
## groups ("modes"). In this example, a strictly heterosexual population
## for this example, assume m1 = females, m2 = males 
num.g1 <- num.g2 <- 500
nw <- network::network.initialize(num.g1 + num.g2, bipartite = num.g1, directed = FALSE)
set.vertex.attribute(nw, "group", rep(1:2, c(num.g1, num.g2)))

## degree distribution: we need the number of partnerships 
##      implied by each distribution to be the same.
## In this example of 1:1 sex ratio, set the mean proportions of persons
##      in each mode with 0, 1, 2, 3 partners at any one time:
deg.dist.g1 <- c(0.40, 0.55, 0.04, 0.01)
deg.dist.g2 <- c(0.48, 0.41, 0.08, 0.03)

## Check the implied number of edges matches across modes
check_degdist_bal(num.g1, num.g2, deg.dist.g1, deg.dist.g2)

##===========================================
## Network model estimation and diagnostics
##===========================================
## specify model
formation <- ~edges + b1degree(0:1) + b2degree(0:1)

## input data as target statistics from the initial
##      network for each of the terms (from the bipartite check table)
target.stats <- c(330, 200, 275, 240, 205)

## Set overall death rate (d.rate)
coef.diss <- dissolution_coefs(dissolution = ~ offset(edges), duration = 25, d.rate = 0.006)
coef.diss

## `netest` function estimates the model
est2 <- netest(nw, formation, target.stats, coef.diss, edapprox = TRUE)

## run model diagnostics before moving on to epidemic simulation
## *newtorks simulated in `netdx` do not include demographic processes*
## so you must run post_simulation diagnostics after the epi model is simulated
dx <- netdx(est2, nsims = 10, nsteps = 1000)

## Network diagnostics plot
## Dashed lines = target statistics, solid lines = mean simulated stats, bands = iqr)
plot(dx, plots.joined = TRUE, legend = "topleft", main = "Network diagnostics plot")

##======================================
## Epidemic model setup and simulation
##======================================
## set initial number of infected to 50 in mode groups
init <- init.net(i.num = 50, i.num.g2 = 50)

## Using different transmission rates by group (with females 3x more susceptible)
## birth rate (a.rate.g2) set to NA for "male" group
## mortality rate (ds.rate.g1) specified lower for females than males
## increased risk of mortality upon infection (di.rate/g2)
param <- param.net(inf.prob = 0.3, inf.prob.g2 = 0.1, a.rate = 0.006, a.rate.g2 = NA, ds.rate = 0.005, ds.rate.g2 = 0.006, di.rate = 0.008, di.rate.g2 = 0.009)

## set to remove inactive nodes at each timestep for computational efficiency
## monitor mean degree for differences from formation formula
control <- control.net(type = "SI", nsims = 10, nsteps = 500, nwstats.formula = ~edges + meandeg, delete.nodes = TRUE)
sim2 <- netsim(est2, param, init, control)

## Plot 1: Number of Edges x time, Plot 2: Mean Degree x time
## mean degree should be preserved over time despite changes in number of edges
plot(sim2, type = "formation", plots.joined = FALSE, legend = TRUE)
abline(h = 0.66, lty = 2, lwd = 2)

## Plot 1: Prevalence x time (iqr suppressed by default)  Plot 2: abs. numbers in each compartment x time 
## overall population size declining substantially due to disease-induced mortality
par(mfrow = c(1, 2), mar = c(3, 3, 1, 1), mgp = c(2, 1, 0))
plot(sim2, popfrac = TRUE)
plot(sim2, popfrac = FALSE)
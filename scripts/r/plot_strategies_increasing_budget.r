library(ggplot2)
library(dplyr)

dat1 <- read.csv("../../results/none_stratified_n2.csv")
dat2 <- read.csv("../../results/aposteriori_stratified_n2.csv")
dat3 <- read.csv("../../results/apriori_stratified_n2.csv")
dat <- rbind(dat1, dat2, dat3)
dat <- dat[dat$agent == 0,]

dat <- dat %>%
  group_by(collaboration,sample,agents,budget,agent) %>%
  summarise(dp_error_mean = mean(dp_error), sd=sd(dp_error))

print(dat)

p <- ggplot(dat, aes(x=budget, y=dp_error_mean, shape=collaboration, group=collaboration, color=collaboration)) +
     geom_line() +
     geom_point() +
     #geom_errorbar(aes(ymin=dp_error_mean-sd, ymax=dp_error_mean+sd), width=.2, position=position_dodge(.9)) +
     theme_bw() +
     theme(legend.position=c(0.7, 0.7), legend.box.background = element_rect(colour = "black")) +
     xlab("Budget") +
     ylab("DP Error")

ggsave("../../results/budget.pdf", p, width=4, height=3)

library(ggplot2)
library(dplyr)

dat <- read.csv("../../results/german_credit_stratified_n2.csv")

dat <- dat %>%
  group_by(collaboration,sample,agents,budget,agent) %>%
  summarise(dp_error_mean = mean(dp_error), sd=sd(dp_error))
print(dat, n=1000)

dat$agent <- factor(dat$agent,
                    levels = unique(dat$agent),
                    labels = paste0("Agent ", unique(dat$agent)))


p <- ggplot(dat, aes(x=budget, y=dp_error_mean, shape=collaboration, group=collaboration, color=collaboration)) +
     geom_line() +
     geom_point() +
     #geom_errorbar(aes(ymin=dp_error_mean-sd, ymax=dp_error_mean+sd), width=.2, position=position_dodge(.9)) +
     theme_bw() +
     theme(legend.position="top", legend.box.background = element_rect(colour = "black")) +
     xlab("Budget") +
     ylab("DP Error") +
     facet_wrap(~ agent)

ggsave("../../results/budget.pdf", p, width=5.7, height=3)

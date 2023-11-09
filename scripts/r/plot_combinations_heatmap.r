library(ggplot2)
library(dplyr)

dat <- read.csv("../../results/combinations_n2_synthetic.csv")

dat <- dat %>%
  group_by(collaboration,sample,agents,budget,agent, a0, a1) %>%
  summarise(dp_error_mean = mean(dp_error), sd=sd(dp_error))
print(dat, n=1000)

p <- ggplot(dat, aes(x=a0, y=a1, fill=dp_error_mean)) +
     geom_tile() +
     geom_text(aes(label = round(dp_error_mean, 2))) +
     xlab("Agent 0 attribute") +
     ylab("Agent 1 attribute") +
     theme_bw() +
     scale_fill_gradient(name="DP Error", low="white", high="red", limits=c(0, 0.1)) +
     theme(legend.position="bottom") +
     guides(fill=guide_legend(label.position = "bottom", keywidth = 3, keyheight = 1)) +
     facet_wrap(~ collaboration)

ggsave("../../results/heatmap.pdf", p, width=6, height=4)

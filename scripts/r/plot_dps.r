library(ggplot2)

dat <- read.csv("../../results/dps.csv")
dat_filtered <- dat[dat$dataset == "synthetic",]

p <- ggplot(dat_filtered, aes(x=attribute, y=dp)) +
     geom_bar(stat="identity") +
     theme_bw() +
     xlab("Attribute") +
     ylab("Demographic Parity")

ggsave("../../results/dps.pdf", p, width=5.7, height=3)

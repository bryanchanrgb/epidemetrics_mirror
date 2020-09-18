library(dplyr)
library(purrr)
library(magrittr)

x <- read.csv("data/2020-09-15/gni_data.csv",
              stringsAsFactors = FALSE) %>%
  select(countrycode, gni_per_capita)

y <- read.csv("data/2020-09-13/fig1a-t0-days.csv",
              stringsAsFactors = FALSE) %>%
  select(countrycode, days_to_t0)


z <- full_join(x = x, y = y, by = "countrycode") %>%
  filter(not(is.na(gni_per_capita)),
         not(is.na(days_to_t0)))

write.table(x = z,
            file = "t0-and-gni.csv",
            sep = ",",
            row.names = FALSE)

sink("figure-1-t0-and-gni.txt")
summary(lm(log(days_to_t0)~log(gni_per_capita), z))
cat("And again removing the outlier\n")
summary(lm(log(days_to_t0)~log(gni_per_capita), filter(z, days_to_t0 > 30)))
sink()

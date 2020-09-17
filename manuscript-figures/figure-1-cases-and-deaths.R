
library(magrittr)
library(dplyr)
library(reshape2)
library(ggplot2)


y <- read.csv("data/2020-09-13/fig1c-t0-epi-state.csv") %>%
  filter(class == 4) %>%
  use_series(countrycode)

x <- read.csv("data/2020-09-13/figure_5.csv") %>%
  select(countrycode,date,new_per_day,dead_per_day) %>%
  filter(is.element(el = countrycode, set = y)) %>%
  mutate(date = as.Date(date)) %>%
  melt(id.vars = c("countrycode", "date"))


writeLines(text = as.character(
             jsonlite::toJSON(lapply(1:nrow(x), function(ix) as.list(x[ix,])),
                              auto_unbox = TRUE,
                              pretty = TRUE)
           ),
           con = "cases-and-deaths-data.json")



bp_datum <- function(class_name, bp_nums, outliers) {
  list(class = class_name,
       lower = bp_nums[1],
       q1 = bp_nums[2],
       median = bp_nums[3],
       q3 = bp_nums[4],
       upper = bp_nums[5],
       outliers = outliers)
}

datum_list <- function(bp_data, ix) {
  class_name <- bp_data$names[ix]
  bp_nums <- bp_data$stats[,ix]
  out_mask <- bp_data$group == ix
  outliers <- bp_data$out[out_mask]
  bp_datum(class_name, bp_nums, outliers)
}


data_json <- function(bp_data) {
  lapply(1:length(bp_data$names), function(ix) datum_list(bp_data, ix))
}


x <- read.csv("data/2020-09-13/fig1c-t0-epi-state.csv")

bp_data <- boxplot(days_to_t0 ~ class, data = x)

json_string <- as.character(jsonlite::toJSON(data_json(bp_data), auto_unbox = TRUE, pretty = TRUE))

writeLines(text = json_string, con = "boxplot-data.json")

# Load Packages, Clear, Sink -------------------------------------------------------

# load packages
package_list <- c("readr","ggplot2","gridExtra","plyr","dplyr","ggsci","RColorBrewer",
                  "viridis","sf","reshape2","ggpubr","egg","scales","plotrix","ggallin", "stats")
for (package in package_list){
  if (!package %in% installed.packages()){
    install.packages(package)
  }
}
lapply(package_list, require, character.only = TRUE)

# clear workspace
rm(list=ls())

# Import Data ------------------------------------------------------------

# Import csv file for Figure 2
figure_2_data <- read_csv("../data/figure_2.csv",
                          na = c("N/A","NA","#N/A"," ",""),
                          col_types = cols(date = col_date(format = "%Y-%m-%d"),
                                           new_tests = col_double(),
                                           new_tests_smooth = col_double(),
                                           positive_rate = col_double(),
                                           positive_rate_smooth = col_double(),
                                           cfr_smooth = col_double()))


# Process Data for Figure 2 ------------------------------------------------

# Set up colour palette
my_palette_1 <- brewer.pal(name="YlGnBu",n=4)[2]
my_palette_2 <- brewer.pal(name="YlGnBu",n=4)[4]
my_palette_3 <- brewer.pal(name="Oranges",n=4)[4]

# Cut the positive rate before April due to low denominator
figure_2_data[figure_2_data$date<'2020-04-01','positive_rate'] <- NA
figure_2_data[figure_2_data$date<'2020-05-01','cfr_smooth'] <- NA
# For italy, need to exclude some dates from CFR calculation when cases were very low
figure_2_data[(figure_2_data$country=='Italy')&(figure_2_data$date>='2020-08-16')&(figure_2_data$date<='2020-08-22'),'cfr_smooth'] <- NA


# Define which countries to plot
country_list = c("Italy","France","United States")

# Plot Figure 2 - with individual subplots and grid arrange ---------------------------------

for (country_a in country_list){
  # Individual subplots
  figure_2_a_1 <- (ggplot(subset(figure_2_data,country==country_a)) 
                   + geom_line(aes(x = date, y = new_per_day),size=0.3,color=my_palette_1,na.rm=TRUE)
                   + geom_line(aes(x = date, y = new_per_day_smooth),color=my_palette_2,na.rm=TRUE)
                   + labs(title=country_a, y="New Cases per Day", x=element_blank())
                   + theme_classic(base_size=8,base_family='serif')
                   + scale_y_continuous(expand = c(0,0),limits = c(0, NA))
                   + theme(plot.title=element_text(size=8, hjust = 0.5),plot.margin=unit(c(0,0,0,0),"pt")))
  max_dead_a=max(subset(figure_2_data,country==country_a)$dead_per_day,na.rm=TRUE)
  max_cfr_a=max(subset(figure_2_data,country==country_a)$cfr_smooth,na.rm=TRUE)
  figure_2_a_2 <- (ggplot(subset(figure_2_data,country==country_a)) 
                   + geom_line(aes(x = date, y = dead_per_day),size=0.3,color=my_palette_1,na.rm=TRUE)
                   + geom_line(aes(x = date, y = dead_per_day_smooth),color=my_palette_2,na.rm=TRUE)
                   + geom_line(aes(x = date, y = cfr_smooth*(max_dead_a/max_cfr_a)),color=my_palette_3,na.rm=TRUE)
                   + scale_y_continuous(name = "Deaths per Day", 
                                        expand = c(0,0),limits = c(0, NA),
                                        sec.axis = sec_axis(~./(max_dead_a/max_cfr_a), name = "Case Fatality Rate"))
                   + theme(plot.title = element_text(hjust = 0.5),plot.margin=unit(c(0,0,0,0),"pt"))
                   + theme_classic(base_size=8,base_family='serif')
                   + theme(plot.title = element_text(hjust = 0.5),plot.margin=unit(c(0,0,0,0),"pt"),
                           axis.title.y.left = element_text(color=my_palette_2), axis.title.y.right = element_text(color=my_palette_3)))
  max_tests_a=max(subset(figure_2_data,country==country_a)$new_tests,na.rm=TRUE)
  max_positive_rate_a=max(subset(figure_2_data,country==country_a)$positive_rate_smooth,na.rm=TRUE)
  figure_2_a_3 <- (ggplot(subset(figure_2_data,country==country_a)) 
                   + geom_line(aes(x = date, y = new_tests),size=0.3,color=my_palette_1,na.rm=TRUE)
                   + geom_line(aes(x = date, y = new_tests_smooth),color=my_palette_2,na.rm=TRUE)
                   + geom_line(aes(x = date, y = positive_rate_smooth*(max_tests_a/max_positive_rate_a)),color=my_palette_3,na.rm=TRUE)
                   + scale_y_continuous(name = "Tests per Day", 
                                        expand = c(0,0),limits = c(0, NA),
                                        sec.axis = sec_axis(~./(max_tests_a/max_positive_rate_a), name = "Positive Rate"))
                   + labs(x="Date")
                   + theme_classic(base_size=8,base_family='serif')
                   + theme(plot.title = element_text(hjust = 0.5),plot.margin=unit(c(0,0,0,0),"pt"),
                           axis.title.y.left = element_text(color=my_palette_2), axis.title.y.right = element_text(color=my_palette_3)))
  # Combining subplots
  figure_2_all <- egg::ggarrange(figure_2_a_1,
                                 figure_2_a_2,
                                 figure_2_a_3,
                                 ncol=1)
  figure_2_all <- annotate_figure(figure_2_all,
                                  top = text_grob("Figure 2: Cases, Deaths and Testing Over Time", size = 9, family='serif'))
  ggsave(paste("../plots/figure_2_",country_a,".png",sep=""), plot = figure_2_all, width = 8,  height = 10, units='cm',dpi=300)
}

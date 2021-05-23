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

# Import csv file for Figure 3 
figure_3_all_data <- read_csv("./data/figure_3_all.csv", 
                                na = c("N/A","NA","#N/A"," ",""))
figure_3_all_data$countrycode = as.factor(figure_3_all_data$countrycode)
figure_3_all_data$country = as.factor(figure_3_all_data$country)
figure_3_all_data$class = as.factor(figure_3_all_data$class)

# Process Data ----------------------------------------------

# Normalise by population size
figure_3_all_data$dead_during_wave_per_10k <- figure_3_all_data$dead_during_wave * 10000 / figure_3_all_data$population
figure_3_all_data$tests_during_wave_per_10k <- figure_3_all_data$tests_during_wave * 10000 / figure_3_all_data$population

# Calculate response time as date to reach threshold - date of T0
figure_3_all_data$c3_response_time <- figure_3_all_data$first_date_c3_above_threshold - figure_3_all_data$t0_10_dead
figure_3_all_data$c3_response_time = as.numeric(figure_3_all_data$c3_response_time)
figure_3_all_data$testing_response_time <- figure_3_all_data$first_date_tests_above_threshold - figure_3_all_data$t0_10_dead
figure_3_all_data$testing_response_time = as.numeric(figure_3_all_data$testing_response_time)

# Highlight some countries
highlight_countries = c('Australia','Belgium','United States')
figure_3_all_data$country_highlight = ''
for (c in highlight_countries){
  figure_3_all_data[figure_3_all_data$country==c,'country_highlight'] <- c
}
figure_3_all_data$country_highlight <- factor(figure_3_all_data$country_highlight, levels=c('Australia','Belgium','United States',''))

# Remove very small countries as their T0 are skewed
figure_3_data <- subset(figure_3_all_data,population>=2500000)

# Countries to label in scatterplot --------------------------------------
label_countries <- c("USA","GBR","ESP","BRA","JAP","IND","ZAF","BEL","AUS")

my_palette_1 = c()
for (i in c(2,3,4)){my_palette_1[i] <- brewer.pal(name="YlGnBu",n=5)[4]}
my_palette_1[1] <- '#000000'


# Plot Figure 3 ------------------------------------------------------------

plot_figure <- function(data,y,x,y_title,x_title,y_trans='identity',x_trans='identity') {
  data = figure_3_data[figure_3_data[['wave']] == wave, ]
  corr <- cor.test(data[[x]], data[[y]], method = "kendall")
  p_value_str <- if (corr$p.value<0.0001) {"<0.0001"} else {toString(signif(corr$p.value,2))}
  estimate_str <- toString(signif(corr$estimate,2))
  corr_text <- paste("Kendall's Rank Correlation \nTau Estimate: ",estimate_str," \np-value: ",p_value_str,sep="")
  
  label_data = data[data[['countrycode']] %in% label_countries |
                      data[['countrycode']] %in% highlight_countries |
                      data[[x]] >= quantile(data[[x]], 0.95,na.rm=TRUE) |
                      data[[x]] <= quantile(data[[x]], 0.05,na.rm=TRUE), ]
  
  figure <- (ggplot(data, aes(x = data[[x]], y = data[[y]], color=country_highlight)) 
             + geom_point(size=1, na.rm=TRUE, show.legend = FALSE)
             + geom_text(data=label_data, aes(x = label_data[[x]], y = label_data[[y]],label=country), hjust=-0.1, vjust=-0.5, size=2.2, family='serif',show.legend = FALSE)
             + geom_text(aes(x=0,y=0,label=corr_text),size=2.4, hjust=-0.5, vjust=-0.5, family='serif',color='black')
             + theme_classic(base_size=8,base_family='serif')
             + theme(plot.title=element_text(size = 8, hjust = 0.5))
             + scale_color_manual(values = my_palette_1, name = "Country")
             + scale_x_continuous(trans=x_trans, expand=expand_scale(mult=c(0,0.2)))
             + scale_y_continuous(trans=y_trans)
             + labs(title = paste(y_title," Against ",x_title,sep=""), x = x_title, y = y_title))
  if (wave == 1) {y_text = sub('during_wave','during_first_wave',y)} 
  else if (wave == 2) {y_text = sub('during_wave','during_second_wave',y)} 
  else {y_text = y}
  ggsave(paste("./plots/figure_3_",y_text,"_",x,".png",sep=""), plot=figure, width=10, height=10, units='cm', dpi=300)
}

# deaths in first wave vs. si in first wave
wave <- 1
y <- 'dead_during_wave_per_10k'
x <- 'si_integral_during_wave'
y_title <- "Deaths During First Wave per 10000"
x_title <- "Stringency During First Wave"
y_trans <- 'log10'
x_trans <- 'identity'
plot_figure(data,y,x,y_title,x_title,y_trans,x_trans)


# deaths in first wave vs. si in all waves
wave <- 1
y <- 'dead_during_wave_per_10k'
x <- 'si_integral'
y_title <- "Deaths During First Wave per 10000"
x_title <- "Total Stringency Across All Waves"
y_trans <- 'log10'
x_trans <- 'identity'
plot_figure(data,y,x,y_title,x_title,y_trans,x_trans)


# deaths in first wave vs. c3 response time
wave <- 1
y <- 'dead_during_wave_per_10k'
x <- 'c3_response_time'
y_title <- "Deaths During First Wave per 10000"
x_title <- "Response Time (Days from T0 to Cancelling Public Events)"
y_trans <- 'log10'
x_trans <- pseudolog10_trans
plot_figure(data,y,x,y_title,x_title,y_trans,x_trans)


# deaths in first wave vs. tests in first wave
wave <- 1
y <- 'dead_during_wave_per_10k'
x <- 'tests_during_wave_per_10k'
y_title <- "Deaths During First Wave per 10000"
x_title <- "Tests During First Wave per 10000"
y_trans <- 'log10'
x_trans <- 'log10'
plot_figure(data,y,x,y_title,x_title,y_trans,x_trans)

# deaths in first wave vs. tests in all waves
wave <- 1
y <- 'dead_during_wave_per_10k'
x <- 'last_tests_per_10k'
y_title <- "Deaths During First Wave per 10000"
x_title <- "Total Tests per 10000"
y_trans <- 'log10'
x_trans <- 'log10'
plot_figure(data,y,x,y_title,x_title,y_trans,x_trans)

# deaths in first wave vs. testing response time
wave <- 1
y <- 'dead_during_wave_per_10k'
x <- 'testing_response_time'
y_title <- "Deaths During First Wave per 10000"
x_title <- "Response Time (Days from T0 to 10 Tests per 10000)"
y_trans <- 'log10'
x_trans <- pseudolog10_trans
plot_figure(data,y,x,y_title,x_title,y_trans,x_trans)


# deaths in second wave vs. si in second wave
wave <- 2
y <- 'dead_during_wave_per_10k'
x <- 'si_integral_during_wave'
y_title <- "Deaths During Second Wave per 10000"
x_title <- "Stringency During Second Wave"
y_trans <- 'log10'
x_trans <- 'identity'
plot_figure(data,y,x,y_title,x_title,y_trans,x_trans)

# deaths in second wave vs. si in all waves
wave <- 2
y <- 'dead_during_wave_per_10k'
x <- 'si_integral'
y_title <- "Deaths During Second Wave per 10000"
x_title <- "Total Stringency Across All Waves"
y_trans <- 'log10'
x_trans <- 'identity'
plot_figure(data,y,x,y_title,x_title,y_trans,x_trans)


# deaths in second wave vs. tests in second wave
wave <- 2
y <- 'dead_during_wave_per_10k'
x <- 'tests_during_wave_per_10k'
y_title <- "Deaths During Second Wave per 10000"
x_title <- "Tests During Second Wave per 10000"
y_trans <- 'log10'
x_trans <- 'log10'
plot_figure(data,y,x,y_title,x_title,y_trans,x_trans)


# deaths in second wave vs. tests in all waves
wave <- 2
y <- 'dead_during_wave_per_10k'
x <- 'last_tests_per_10k'
y_title <- "Deaths During Second Wave per 10000"
x_title <- "Total Tests per 10000"
y_trans <- 'log10'
x_trans <- 'log10'
plot_figure(data,y,x,y_title,x_title,y_trans,x_trans)


# deaths in total vs. si in first wave
wave <- 1
y <- 'last_dead_per_10k'
x <- 'si_integral_during_wave'
y_title <- "Total Deaths per 10000"
x_title <- "Stringency During First Wave"
y_trans <- 'log10'
x_trans <- 'identity'
plot_figure(data,y,x,y_title,x_title,y_trans,x_trans)

# deaths in total vs. si in second wave
wave <- 2
y <- 'last_dead_per_10k'
x <- 'si_integral_during_wave'
y_title <- "Total Deaths per 10000"
x_title <- "Stringency During Second Wave"
y_trans <- 'log10'
x_trans <- 'identity'
plot_figure(data,y,x,y_title,x_title,y_trans,x_trans)

# deaths in total vs. si in all waves
wave <- 1
y <- 'last_dead_per_10k'
x <- 'si_integral'
y_title <- "Total Deaths per 10000"
x_title <- "Stringency Across All Waves"
y_trans <- 'log10'
x_trans <- 'identity'
plot_figure(data,y,x,y_title,x_title,y_trans,x_trans)

# deaths in total vs. c3 response time
wave <- 1
y <- 'last_dead_per_10k'
x <- 'c3_response_time'
y_title <- "Total Deaths per 10000"
x_title <- "Response Time (Days from T0 to Cancelling Public Events)"
y_trans <- 'log10'
x_trans <- pseudolog10_trans
plot_figure(data,y,x,y_title,x_title,y_trans,x_trans)


# deaths in total vs. tests in first wave
wave <- 1
y <- 'last_dead_per_10k'
x <- 'tests_during_wave_per_10k'
y_title <- "Total Deaths per 10000"
x_title <- "Tests During First Wave per 10000"
y_trans <- 'log10'
x_trans <- 'log10'
plot_figure(data,y,x,y_title,x_title,y_trans,x_trans)

# deaths in total vs. tests in second wave
wave <- 2
y <- 'last_dead_per_10k'
x <- 'tests_during_wave_per_10k'
y_title <- "Total Deaths per 10000"
x_title <- "Tests During Second Wave per 10000"
y_trans <- 'log10'
x_trans <- 'log10'
plot_figure(data,y,x,y_title,x_title,y_trans,x_trans)

# deaths in total vs. tests in all waves
wave <- 1
y <- 'last_dead_per_10k'
x <- 'last_tests_per_10k'
y_title <- "Total Deaths per 10000"
x_title <- "Total Tests per 10000"
y_trans <- 'log10'
x_trans <- 'log10'
plot_figure(data,y,x,y_title,x_title,y_trans,x_trans)

# deaths in total vs. testing response time
wave <- 1
y <- 'last_dead_per_10k'
x <- 'testing_response_time'
y_title <- "Total Deaths per 10000"
x_title <- "Response Time (Days from T0 to 10 Tests per 10000)"
y_trans <- 'log10'
x_trans <- pseudolog10_trans
plot_figure(data,y,x,y_title,x_title,y_trans,x_trans)


# replace nulls in second wave with 0
country_2 <- figure_3_all_data[figure_3_all_data$wave==2,'countrycode']
data_2 <- figure_3_all_data[figure_3_all_data$wave==2,]
data_1 <- figure_3_all_data[(figure_3_all_data$wave==1)&!(figure_3_all_data$countrycode%in%country_2$countrycode),]
data_1$wave <- 2
data_1$dead_during_wave <- 0
data_1$dead_during_wave_per_10k <- 0
data_1$tests_during_wave <- 0
data_1$tests_during_wave_per_10k <- 0
data_1$si_integral_during_wave <- 0
figure_3_data <- rbind(data_1,data_2)

# deaths during second wave against stringency response time 
wave <- 2
y <- 'dead_during_wave_per_10k'
x <- 'c3_response_time'
y_title <- "Deaths in Second Wave per 10000"
x_title <- "Response Time (Days from T0 to Cancelling Public Events)"
y_trans <- 'log10'
x_trans <- pseudolog10_trans
plot_figure(data,y,x,y_title,x_title,y_trans,x_trans)


# deaths during second wave against testing response time 
wave <- 2
y <- 'dead_during_wave_per_10k'
x <- 'testing_response_time'
y_title <- "Deaths in Second Wave per 10000"
x_title <- "Response Time (Days from T0 to 10 Tests per 10000)"
y_trans <- 'log10'
x_trans <- pseudolog10_trans
plot_figure(data,y,x,y_title,x_title,y_trans,x_trans)


figure_3_data <- figure_3_all_data[figure_3_all_data$wave==2,]

# likelihood of second wave given response time
data <- figure_3_all_data[figure_3_all_data$wave==1,c('countrycode','c3_response_time','testing_response_time')]
country_2 <- figure_3_all_data[figure_3_all_data$wave==2,'countrycode']
data$second_wave <- data$countrycode%in%country_2$countrycode
data[data$second_wave==TRUE,'second_wave'] <- 1
data[data$second_wave==FALSE,'second_wave'] <- 0
mylogit <- glm(second_wave ~ c3_response_time, data = data, family = "binomial")
summary(mylogit)

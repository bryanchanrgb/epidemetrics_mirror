# Epidemetrics - Generate Plots (Figures 2, 3)

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

# Set working directory

# Import Data ------------------------------------------------------------

# Import csv file for Figure 2
figure_2_data <- read_csv("./data/figure_2.csv", 
                          na = c("N/A","NA","#N/A"," ",""),
                          col_types = cols(date = col_date(format = "%Y-%m-%d"),
                                           new_tests = col_double(),
                                           new_tests_smooth = col_double(),
                                           positive_rate = col_double(),
                                           positive_rate_smooth = col_double()))

# Import csv file for Figure 3a
figure_3a_data <- read_csv("./data/figure_3a.csv", 
                           na = c("N/A","NA","#N/A"," ",""))
figure_3a_data$countrycode = as.factor(figure_3a_data$countrycode)
figure_3a_data$country = as.factor(figure_3a_data$country)
figure_3a_data$class = as.factor(figure_3a_data$class)

# Import csv file for Figure 3b
figure_3b_data <- read_csv("./data/figure_3b.csv", 
                           na = c("N/A","NA","#N/A"," ",""))
figure_3b_data$COUNTRYCODE = as.factor(figure_3b_data$COUNTRYCODE)
figure_3b_data$COUNTRY = as.factor(figure_3b_data$COUNTRY)
figure_3b_data$CLASS = as.factor(figure_3b_data$CLASS)
figure_3b_data$CLASS_COARSE = as.factor(figure_3b_data$CLASS_COARSE)

# Import csv file for Figure 3b at the wave level
figure_3b_wave_data <- read_csv("./data/figure_3b_wave_level.csv", 
                           na = c("N/A","NA","#N/A"," ",""))
figure_3b_wave_data$countrycode = as.factor(figure_3b_wave_data$countrycode)
figure_3b_wave_data$country = as.factor(figure_3b_wave_data$country)
figure_3b_wave_data$class = as.factor(figure_3b_wave_data$class)
figure_3b_wave_data$wave = as.factor(figure_3b_wave_data$wave)

# Import csv file for Figure 3c at the wave level
figure_3c_1_data <- read_csv("./data/figure_3c_1.csv", 
                                na = c("N/A","NA","#N/A"," ",""))
figure_3c_1_data$countrycode = as.factor(figure_3c_1_data$countrycode)
figure_3c_1_data$country = as.factor(figure_3c_1_data$country)
figure_3c_1_data$class = as.factor(figure_3c_1_data$class)
figure_3c_1_data$wave = as.factor(figure_3c_1_data$wave)

# Import csv file for Figure 3c testing data
figure_3c_2_data <- read_csv("./data/figure_3c_2.csv", 
                             na = c("N/A","NA","#N/A"," ",""))
figure_3c_2_data$countrycode = as.factor(figure_3c_2_data$countrycode)


# Countries to label in scatterplot --------------------------------------
label_countries <- c("USA","GBR","ESP","BRA","JAP","IND","ZAF","BEL","AUS")


# Process Data for Figure 3 ----------------------------------------------

# Remove Others class from data. Only keep classes 1-4 for now, 5 has low sample size
#figure_3b_data <- subset(figure_3b_data,CLASS%in%c(1,2,3,4))

# Reorder factor levels
figure_3a_data$class_coarse <- factor(figure_3a_data$class_coarse, levels=c("EPI_FIRST_WAVE","EPI_SECOND_WAVE","EPI_THIRD_WAVE","EPI_OTHER"))

# Process data for 3b: remove all second wave observations
figure_3b_wave_1_data <- subset(figure_3b_wave_data,wave==1)
figure_3b_wave_1_data <- subset(figure_3b_wave_1_data,country!="Qatar") # Qatar is badly classified
# Calculate response time as date to reach threshold - date of T0
figure_3b_wave_1_data$response_time <- figure_3b_wave_1_data$first_date_c1_above_threshold - figure_3b_wave_1_data$t0_10_dead
figure_3b_wave_1_data$response_time = as.numeric(figure_3b_wave_1_data$response_time)
# Remove any where the date SI reaches threshold is no longer during the first wave
figure_3b_wave_1_data <- subset(figure_3b_wave_1_data,first_date_si_above_threshold<wave_end)
# Remove very small countries as their T0 are skewed
figure_3b_wave_1_data <- subset(figure_3b_wave_1_data,population>=2500000)

figure_3b_wave_data$dead_during_wave_per_10k <- figure_3b_wave_data$dead_during_wave * 10000 / figure_3b_wave_data$population
figure_3b_wave_1_data$dead_during_wave_per_10k <- figure_3b_wave_1_data$dead_during_wave * 10000 / figure_3b_wave_1_data$population


# Plot Figure 3a ------------------------------------------------------------
# Figure 3: Scatter plot of government response time against number of cases for each country
corr <- cor.test(figure_3a_data$si_integral, figure_3a_data$last_dead_per_10k,
         method = "kendall")
p_value_str <- if (corr$p.value<0.0001) {"<0·0001"} else {sub('[.]','·',toString(signif(corr$p.value,2)))}
estimate_str <- sub('[.]','·',toString(signif(corr$estimate,2)))
corr_text <- paste("Kendall's Rank Correlation \nTau Estimate: ",estimate_str," \np-value: ",p_value_str,sep="")
figure_3a <- (ggplot(figure_3a_data, aes(x = si_integral, y = last_dead_per_10k)) 
              + geom_point(size=2,alpha=0.9, na.rm=TRUE)
              + geom_text(data=subset(figure_3a_data,
                                      ((countrycode %in% label_countries) |
                                        #(last_dead_per_10k >= quantile(figure_3a_data$last_dead_per_10k, 0.95,na.rm=TRUE)) |
                                        (si_integral >= quantile(figure_3a_data$si_integral, 0.95,na.rm=TRUE)) |
                                        (si_integral <= quantile(figure_3a_data$si_integral, 0.05,na.rm=TRUE))) &
                                        (!country %in% c('Eritrea','Cayman Islands','United Kingdom','United States','Brazil','Bolivia','South Africa'))),
                          aes(label=country),
                          hjust=-0.1, vjust=-0.1,
                          show.legend = FALSE)
              + geom_text(aes(x=3000,y=10,hjust=0,label=corr_text),size=4, hjust=0, color='black')
              + theme_light()
              + theme(plot.title=element_text(hjust = 0.5), axis.line=element_line(color="black",size=0.7),axis.ticks=element_line(color="black",size=0.7), legend.position=c(0.85, 0.15))
              #+ scale_color_discrete(name = "Wave Status", labels = c("First Wave", "Second Wave","Third Wave","Other"))
              + scale_x_continuous(expand=expand_scale(mult=c(0.05,0.12)))
              + scale_y_continuous(trans='log10', breaks=c(0.001,0.003,0.01,0.03,0.1,0.3,1,3,10), labels=c('0·001','0·003','0·01','0·03','0·1','0·3','1','3','10'))
              + labs(title = "Total Deaths Against Total Government Stringency", x = "Integral Under Stringency Index Curve", y = "Total Deaths per 10000 Population"))
figure_3a
ggsave("./plots/figure_3a.png", plot = figure_3a, width = 9,  height = 7)
ggsave("./plots/figure_3a.pdf", plot = figure_3a, width = 9,  height = 7)


# Plot Figure 3b ------------------------------------------------------------

corr <- cor.test(figure_3b_wave_1_data$dead_during_wave_per_10k, figure_3b_wave_1_data$response_time,
                 method = "kendall")
p_value_str <- if (corr$p.value<0.0001) {"<0·0001"} else {sub('[.]','·',toString(signif(corr$p.value,2)))}
estimate_str <- sub('[.]','·',toString(signif(corr$estimate,2)))
corr_text <- paste("Kendall's Rank Correlation \nTau Estimate: ",estimate_str," \np-value: ",p_value_str,sep="")
# Plot Figure 3: Scatter plot of government response time against number of cases for each country
figure_3b_wave_1 <- (ggplot(figure_3b_wave_1_data, aes(x = response_time, y = dead_during_wave_per_10k)) 
              + geom_point(size=2,alpha=0.9, na.rm=TRUE)
              + geom_text(data=subset(figure_3b_wave_1_data,
                                      ((countrycode %in% label_countries) |
                                      (response_time >= quantile(figure_3b_wave_1_data$response_time, 0.8,na.rm=TRUE)) |
                                      (response_time <= quantile(figure_3b_wave_1_data$response_time, 0.02,na.rm=TRUE))) &
                                      !country %in% c('Philippines','Iran','India','Hungary','Turkey','Dominican Republic')),
                          aes(label=country),
                          hjust=-0.1, vjust=-0.1,
                          show.legend = FALSE)
              + geom_text(aes(x=1,y=0.01,hjust=0,label=corr_text),size=4, hjust=0, color='black')
              + theme_light()
              + theme(plot.title=element_text(hjust = 0.5), axis.line=element_line(color="black",size=0.7),axis.ticks=element_line(color="black",size=0.7), legend.position=c(0.6, 0.15))
              #+ scale_color_manual(values = my_palette_1, name = "Wave", labels = c("First Wave", "Second Wave"))
              + scale_x_continuous(trans=pseudolog10_trans, breaks=c(-200,-100,-30,-10,-3,-1,0,1,3,10,30), expand=expand_scale(mult=c(0.05,0.2)))
              + scale_y_continuous(trans='log10', breaks=c(0.001,0.01,0.1,1,10),labels=c('0·001','0·01','0·1','1','10'))
              + labs(title = "Total Deaths During First Wave Against Government Response Time", x = "Government Response Time (Days from 10th Death to Schools Closing)", y = "Total Deaths During First Wave per 10000 Population"))
figure_3b_wave_1
ggsave("./plots/figure_3b_wave_1.png", plot = figure_3b_wave_1, width = 9,  height = 7)
ggsave("./plots/figure_3b_wave_1.pdf", plot = figure_3b_wave_1, width = 9,  height = 7)


# Combine figure_3a, 3b
figure_3_all <- grid.arrange(grobs=list(figure_3a,figure_3b,figure_3b),
                             widths = c(1.1, 1),
                             layout_matrix = rbind(c(1, 3),
                                                   c(2,  3)))
figure_3_all <- annotate_figure(figure_3_all,
                                top = text_grob("Figure 3: Government and Public Response", size = 14))
ggsave("./plots/figure_3.png", plot = figure_3_all, width = 15,  height = 8)


# Plot Figure 3c --------------------------------------------------------------


# Process data for 3c: remove all second wave observations
figure_3c_wave_1_data <- subset(figure_3c_1_data,wave==1)
figure_3c_wave_1_data <- subset(figure_3c_wave_1_data,country!="Qatar") # Qatar is badly classified

# Remove very small countries as their T0 are skewed
figure_3c_wave_1_data <- subset(figure_3c_wave_1_data,population>=2500000)

# Add column for tests per population
figure_3c_2_data <- merge(figure_3c_2_data, figure_3c_wave_1_data[c('countrycode','population')], by='countrycode')
# Get per population values
figure_3c_wave_1_data$dead_during_wave_per_10k <- figure_3c_wave_1_data$dead_during_wave * 10000 / figure_3c_wave_1_data$population
figure_3c_2_data$tests_per_10k <- figure_3c_2_data$tests * 10000 / figure_3c_2_data$population


lags <- data.frame(lag=NULL,tau_estimate=NULL,p_value=NULL)
for (lag in seq(-50,100,by=1)){
  if (lag < 0){plus_minus='Before'} else {plus_minus='After'}
  figure_3c_wave_1_data$date <- figure_3c_wave_1_data$t0_10_dead + lag
  plot_data <- merge(figure_3c_wave_1_data, figure_3c_2_data, by=c("countrycode","date"))
  plot_data <- subset(plot_data, date<wave_end) # Remove if lagged date is after the end of the first wave
  corr <- cor.test(plot_data$dead_during_wave_per_10k, plot_data$tests_per_10k,
                   method = "kendall")
  p_value_str <- if (corr$p.value<0.0001) {"<0·0001"} else {sub('[.]','·',toString(signif(corr$p.value,2)))}
  estimate_str <- sub('[.]','·',toString(signif(corr$estimate,2)))
  corr_text <- paste("Kendall's Rank Correlation \nTau Estimate: ",estimate_str," \np-value: ",p_value_str,sep="")
  my_list <- list(lag = lag, tau_estimate = corr$estimate, p_value=corr$p.value)
  lags <- bind_rows(lags, my_list)
  
  # Plot Figure 3c: Scatter plot of tests against number of cases for each country
  figure_3c <- (ggplot(plot_data, aes(x = tests_per_10k, y = dead_during_wave_per_10k)) 
                       + geom_point(size=2,alpha=0.9, na.rm=TRUE)
                       + geom_text(data=subset(plot_data,
                                               (countrycode %in% label_countries) |
                                                 (tests_per_10k >= quantile(plot_data$tests_per_10k, 0.85,na.rm=TRUE)) |
                                                 (tests_per_10k <= quantile(plot_data$tests_per_10k, 0.15,na.rm=TRUE))),
                                   aes(label=country),
                                   hjust=-0.1, vjust=-0.1,
                                   show.legend = FALSE)
                       + geom_text(aes(x=0.02,y=0.02,hjust=0,label=corr_text),size=4, hjust=0, color='black')
                       + theme_light()
                       + theme(plot.title=element_text(hjust = 0.5), axis.line=element_line(color="black",size=0.7),axis.ticks=element_line(color="black",size=0.7), legend.position=c(0.6, 0.15))
                       + scale_x_continuous(trans='log10', breaks=c(0.0001,0.001,0.01,0.1,1,10,100,1000),labels=c('0·0001','0·001','0·01','0·1','1','10','100','1000'), expand=expand_scale(mult=c(0.05,0.05)))
                       + scale_y_continuous(trans='log10', breaks=c(0.001,0.01,0.1,1,10),labels=c('0·001','0·01','0·1','1','10'), expand=expand_scale(mult=c(0.05,0.05)))
                       + labs(title = "Deaths During First Wave Against Early Testing", 
                              x = paste("Total Tests per 10000 Population Up Until",abs(lag),"Days",plus_minus,"the 10th Death",sep=" "),
                              y = "Total Deaths During First Wave per 10000 Population"))
  #figure_3c
  ggsave(paste("./plots/figure_3c_test_lags/figure_3c_",lag,".png",sep=""), plot = figure_3c, width = 9,  height = 7)
}

# Plot Figure 3c: -21 day lag
lag = -21
if (lag < 0){plus_minus='Before'} else {plus_minus='After'}
figure_3c_wave_1_data$date <- figure_3c_wave_1_data$t0_10_dead + lag
plot_data <- merge(figure_3c_wave_1_data, figure_3c_2_data, by=c("countrycode","date"))
plot_data <- subset(plot_data, date<wave_end) # Remove if lagged date is after the end of the first wave
corr <- cor.test(plot_data$dead_during_wave_per_10k, plot_data$tests_per_10k,
                 method = "kendall")
p_value_str <- if (corr$p.value<0.0001) {"<0·0001"} else {sub('[.]','·',toString(signif(corr$p.value,2)))}
estimate_str <- sub('[.]','·',toString(signif(corr$estimate,2)))
corr_text <- paste("Kendall's Rank Correlation \nTau Estimate: ",estimate_str," \np-value: ",p_value_str,sep="")
figure_3c <- (ggplot(plot_data, aes(x = tests_per_10k, y = dead_during_wave_per_10k)) 
              + geom_point(size=2,alpha=0.9, na.rm=TRUE)
              + geom_text(data=subset(plot_data,
                                      ((countrycode %in% label_countries) |
                                        (tests_per_10k >= quantile(plot_data$tests_per_10k, 0.85,na.rm=TRUE)) |
                                        (tests_per_10k <= quantile(plot_data$tests_per_10k, 0.15,na.rm=TRUE))) &
                                        !country %in% c('Uganda','Peru')),
                          aes(label=country),
                          hjust=-0.1, vjust=-0.1,
                          show.legend = FALSE)
              + geom_text(aes(x=0.02,y=0.02,hjust=0,label=corr_text),size=4, hjust=0, color='black')
              + theme_light()
              + theme(plot.title=element_text(hjust = 0.5), axis.line=element_line(color="black",size=0.7),axis.ticks=element_line(color="black",size=0.7), legend.position=c(0.6, 0.15))
              + scale_x_continuous(trans='log10', breaks=c(0.0001,0.001,0.01,0.1,1,10,100,1000),labels=c('0·0001','0·001','0·01','0·1','1','10','100','1000'), expand=expand_scale(mult=c(0.05,0.2)))
              + scale_y_continuous(trans='log10', breaks=c(0.001,0.01,0.1,1,10),labels=c('0·001','0·01','0·1','1','10'), expand=expand_scale(mult=c(0.05,0.05)))
              + labs(title = "Deaths During First Wave Against Early Testing", 
                     x = paste("Total Tests per 10000 Population Up Until",abs(lag),"Days",plus_minus,"the 10th Death",sep=" "),
                     y = "Total Deaths During First Wave per 10000 Population"))
figure_3c
ggsave(paste("./plots/figure_3c_",lag,".png",sep=""), plot = figure_3c, width = 9,  height = 7)
ggsave(paste("./plots/figure_3c_",lag,".pdf",sep=""), plot = figure_3c, width = 9,  height = 7)


# Plot Figure 3c: +70 days
lag = 70
if (lag < 0){plus_minus='Before'} else {plus_minus='After'}
figure_3c_wave_1_data$date <- figure_3c_wave_1_data$t0_10_dead + lag
plot_data <- merge(figure_3c_wave_1_data, figure_3c_2_data, by=c("countrycode","date"))
plot_data <- subset(plot_data, date<wave_end) # Remove if lagged date is after the end of the first wave
corr <- cor.test(plot_data$dead_during_wave_per_10k, plot_data$tests_per_10k,
                 method = "kendall")
p_value_str <- if (corr$p.value<0.0001) {"<0·0001"} else {sub('[.]','·',toString(signif(corr$p.value,2)))}
estimate_str <- sub('[.]','·',toString(signif(corr$estimate,2)))
corr_text <- paste("Kendall's Rank Correlation \nTau Estimate: ",estimate_str," \np-value: ",p_value_str,sep="")
figure_3c <- (ggplot(plot_data, aes(x = tests_per_10k, y = dead_during_wave_per_10k)) 
              + geom_point(size=2,alpha=0.9, na.rm=TRUE)
              + geom_text(data=subset(plot_data,
                                      ((countrycode %in% label_countries) |
                                         (tests_per_10k >= quantile(plot_data$tests_per_10k, 0.85,na.rm=TRUE)) |
                                         (tests_per_10k <= quantile(plot_data$tests_per_10k, 0.15,na.rm=TRUE))) &
                                        !country %in% c('Madagascar','Kenya','Lithuania','United States','United Arab Emirates','Mexico')),
                          aes(label=country),
                          hjust=-0.1, vjust=-0.1,
                          show.legend = FALSE)
              + geom_text(aes(x=0.02,y=0.02,hjust=0,label=corr_text),size=4, hjust=0, color='black')
              + theme_light()
              + theme(plot.title=element_text(hjust = 0.5), axis.line=element_line(color="black",size=0.7),axis.ticks=element_line(color="black",size=0.7), legend.position=c(0.6, 0.15))
              + scale_x_continuous(trans='log10', breaks=c(0.0001,0.001,0.01,0.1,1,10,100,1000),labels=c('0·0001','0·001','0·01','0·1','1','10','100','1000'), expand=expand_scale(mult=c(0.05,0.2)))
              + scale_y_continuous(trans='log10', breaks=c(0.001,0.01,0.1,1,10),labels=c('0·001','0·01','0·1','1','10'), expand=expand_scale(mult=c(0.05,0.05)))
              + labs(title = "Deaths During First Wave Against Late Testing", 
                     x = paste("Total Tests per 10000 Population Up Until",abs(lag),"Days",plus_minus,"the 10th Death",sep=" "),
                     y = "Total Deaths During First Wave per 10000 Population"))
figure_3c
ggsave(paste("./plots/figure_3c_",lag,".png",sep=""), plot = figure_3c, width = 9,  height = 7)
ggsave(paste("./plots/figure_3c_",lag,".pdf",sep=""), plot = figure_3c, width = 9,  height = 7)


# Plot Figure 3c correlation over time as the lag changes
figure_3c_corr_tau <- (ggplot(lags, aes(x = lag, y = tau_estimate)) 
              + geom_point(size=1.5,shape=1,alpha=0.9,stroke=1.5, na.rm=TRUE)
              + theme_light()
              + theme(plot.title=element_text(hjust = 0.5), axis.line=element_line(color="black",size=0.7),axis.ticks=element_line(color="black",size=0.7), legend.position=c(0.6, 0.15))
              + labs(title = "Correlation Between Early Testing and Deaths - Kendall's Tau Estimate at Different Time Periods", 
                     x = "Lag in Days",
                     y = "Kendall's Tau Estimate"))
figure_3c_corr_tau
ggsave("./plots/figure_3c_corr_tau.png", plot = figure_3c_corr_tau, width = 9,  height = 7)
figure_3c_corr_p <- (ggplot(lags, aes(x = lag, y = p_value)) 
                       + geom_point(size=1.5,shape=1,alpha=0.9,stroke=1.5, na.rm=TRUE)
                       + theme_light()
                       + theme(plot.title=element_text(hjust = 0.5), axis.line=element_line(color="black",size=0.7),axis.ticks=element_line(color="black",size=0.7), legend.position=c(0.6, 0.15))
                       + labs(title = "Correlation Between Early Testing and Deaths - Kendall's Tau p-Value at Different Time Periods", 
                              x = "Lag in Days",
                              y = "Kendall's Tau p-Value"))
figure_3c_corr_p
ggsave("./plots/figure_3c_corr_p.png", plot = figure_3c_corr_p, width = 9,  height = 7)


# Ideas and experimenting  -------------------------------------------------------------

# Checking threshold of population to filter out for T0
plot <- (ggplot(figure_3b_wave_1_data, aes(x=t0_10_dead, y=population))
         + geom_point()
         +coord_cartesian(ylim=c(0,3000000))
         + geom_text(aes(label=country), hjust=-0.1, vjust=-0.1,show.legend = FALSE))
plot
ggsave("./plots/ideas/population_low.png", plot=plot, width = 9,  height = 7)
# Many countries with a high population have a very genuinely late T0
# Makes the most sense to just exclude Belize, Guam and Aruba: < 500,000

# Check fast vs slow responders
figure_3b_wave_1_data$fast_responder <- figure_3b_wave_1_data$response_time < -10
figure_3a_data <- merge(figure_3a_data, figure_3b_wave_1_data[c('countrycode','fast_responder')], by = "countrycode", all.x=TRUE)


#-----------------------------------------------------------------------------------------

#-----------------------------------------------------------------------------------

# Process Data for Figure 2 ------------------------------------------------

# Define which countries to plot
country_a = "Australia"
country_b = "Belgium"
country_c = "United States"
country_d = "Malawi"

# Plot Figure 2 - with individual subplots and grid arrange ---------------------------------

# Set up colour palette
my_palette_1 <- brewer.pal(name="YlGnBu",n=4)[2]
my_palette_2 <- brewer.pal(name="YlGnBu",n=4)[4]
my_palette_3 <- brewer.pal(name="Oranges",n=4)[4]

# Cut the positive rate before April due to low denominator
figure_2_data[figure_2_data$date<'2020-04-01','positive_rate'] <- NA

# Individual subplots
figure_2_a_1 <- (ggplot(subset(figure_2_data,country==country_a)) 
                 + geom_line(aes(x = date, y = new_per_day),size=0.7,color=my_palette_1,na.rm=TRUE)
                 + geom_line(aes(x = date, y = new_per_day_smooth),size=1,color=my_palette_2,na.rm=TRUE)
                 + labs(title=country_a, y="New Cases per Day", x=element_blank())
                 + theme_light()
                 + scale_y_continuous(expand = c(0,0),limits = c(0, NA))
                 + theme(plot.title=element_text(size=12, hjust = 0.5), axis.line=element_line(color="black",size=0.7),axis.ticks=element_line(color="black",size=0.7),plot.margin=unit(c(0,0,0,0),"pt")))
figure_2_a_2 <- (ggplot(subset(figure_2_data,country==country_a)) 
                 + geom_line(aes(x = date, y = dead_per_day),size=0.7,color=my_palette_1,na.rm=TRUE)
                 + geom_line(aes(x = date, y = dead_per_day_smooth),size=1,color=my_palette_2,na.rm=TRUE)
                 + labs(y="Deaths per Day",x=element_blank())
                 + scale_y_continuous(expand = c(0,0),limits = c(0, NA))
                 + theme_light()
                 + theme(plot.title = element_text(hjust = 0.5), axis.line=element_line(color="black",size=0.7),axis.ticks=element_line(color="black",size=0.7),plot.margin=unit(c(0,0,0,0),"pt")))

max_tests_a=max(subset(figure_2_data,country==country_a)$new_tests,na.rm=TRUE)
max_positive_rate_a=max(subset(figure_2_data,country==country_a)$positive_rate_smooth,na.rm=TRUE)
figure_2_a_3 <- (ggplot(subset(figure_2_data,country==country_a)) 
                 + geom_line(aes(x = date, y = new_tests),size=0.7,color=my_palette_1,na.rm=TRUE)
                 + geom_line(aes(x = date, y = new_tests_smooth),size=1,color=my_palette_2,na.rm=TRUE)
                 + geom_line(aes(x = date, y = positive_rate_smooth*(max_tests_a/max_positive_rate_a)),color=my_palette_3,na.rm=TRUE)
                 + scale_y_continuous(name = "Tests per Day", 
                                      expand = c(0,0),limits = c(0, NA),
                                      sec.axis = sec_axis(~./(max_tests_a/max_positive_rate_a), name = element_blank(),
                                                          breaks=c(0,0.002,0.004,0.006,0.008,0.010),
                                                          labels=c('0','0·002','0·004','0·006','0·008','0·010')))
                 + labs(x="Date")
                 + theme_light()
                 + theme(plot.title = element_text(hjust = 0.5), axis.line=element_line(color="black",size=0.7),axis.ticks=element_line(color="black",size=0.7),plot.margin=unit(c(0,0,0,0),"pt"),
                         axis.title.y.left = element_text(color=my_palette_2), axis.title.y.right = element_text(color=my_palette_3)))
figure_2_b_1 <- (ggplot(subset(figure_2_data,country==country_b))
                 + geom_line(aes(x = date, y = new_per_day),size=0.7,color=my_palette_1,na.rm=TRUE)
                 + geom_line(aes(x = date, y = new_per_day_smooth),size=1,color=my_palette_2,na.rm=TRUE)
                 + labs(title=country_b,x=element_blank(),y=element_blank())
                 + scale_y_continuous(expand = c(0,0),limits = c(0, NA))
                 + theme_light()
                 + theme(plot.title=element_text(size=12, hjust = 0.5), axis.line=element_line(color="black",size=0.7),axis.ticks=element_line(color="black",size=0.7),plot.margin=unit(c(0,0,0,0),"pt")))
figure_2_b_2 <- (ggplot(subset(figure_2_data,country==country_b)) 
                 + geom_line(aes(x = date, y = dead_per_day),size=0.7,color=my_palette_1,na.rm=TRUE)
                 + geom_line(aes(x = date, y = dead_per_day_smooth),size=1,color=my_palette_2,na.rm=TRUE)
                 + labs(x=element_blank(),y=element_blank())
                 + scale_y_continuous(expand = c(0,0),limits = c(0, NA))
                 + theme_light()
                 + theme(plot.title = element_text(hjust = 0.5), axis.line=element_line(color="black",size=0.7),axis.ticks=element_line(color="black",size=0.7),plot.margin=unit(c(0,0,0,0),"pt")))
max_tests_b=max(subset(figure_2_data,country==country_b)$new_tests,na.rm=TRUE)
max_positive_rate_b=max(subset(figure_2_data,country==country_b)$positive_rate_smooth,na.rm=TRUE)
figure_2_b_3 <- (ggplot(subset(figure_2_data,country==country_b)) 
                 + geom_line(aes(x = date, y = new_tests),size=0.7,color=my_palette_1,na.rm=TRUE)
                 + geom_line(aes(x = date, y = new_tests_smooth),size=1,color=my_palette_2,na.rm=TRUE)
                 + geom_line(aes(x = date, y = positive_rate_smooth*(max_tests_b/max_positive_rate_b)),color=my_palette_3,na.rm=TRUE)
                 + scale_y_continuous(name = element_blank(),
                                      expand = c(0,0),limits = c(0, NA),
                                      sec.axis = sec_axis(~./(max_tests_b/max_positive_rate_b), name = element_blank(),
                                                          breaks=c(0,0.05,0.10,0.15,0.20,0.25,0.30),
                                                          labels=c('0','0·05','0·10','0·15','0·20','0·25','0·30')))
                 + labs(x="Date")
                 + theme_light()
                 + theme(plot.title = element_text(hjust = 0.5), axis.line=element_line(color="black",size=0.7),axis.ticks=element_line(color="black",size=0.7),plot.margin=unit(c(0,0,0,0),"pt"),
                         axis.title.y.left = element_text(color=my_palette_2), axis.title.y.right = element_text(color=my_palette_3)))
figure_2_c_1 <- (ggplot(subset(figure_2_data,country==country_c)) 
                 + geom_line(aes(x = date, y = new_per_day),size=0.7,color=my_palette_1,na.rm=TRUE)
                 + geom_line(aes(x = date, y = new_per_day_smooth),size=1,color=my_palette_2,na.rm=TRUE)
                 + labs(title=country_c,x=element_blank(),y=element_blank())
                 + scale_y_continuous(expand = c(0,0),limits = c(0, NA))
                 + theme_light()
                 + theme(plot.title=element_text(size=12, hjust = 0.5), axis.line=element_line(color="black",size=0.7),axis.ticks=element_line(color="black",size=0.7),plot.margin=unit(c(0,0,0,0),"pt")))
figure_2_c_2 <- (ggplot(subset(figure_2_data,country==country_c))
                 + geom_line(aes(x = date, y = dead_per_day),size=0.7,color=my_palette_1,na.rm=TRUE)
                 + geom_line(aes(x = date, y = dead_per_day_smooth),size=1,color=my_palette_2,na.rm=TRUE)
                 + labs(x=element_blank(),y=element_blank())
                 + scale_y_continuous(expand = c(0,0),limits = c(0, NA))
                 + theme_light()
                 + theme(plot.title = element_text(hjust = 0.5), axis.line=element_line(color="black",size=0.7),axis.ticks=element_line(color="black",size=0.7),plot.margin=unit(c(0,0,0,0),"pt")))
max_tests_c=max(subset(figure_2_data,country==country_c)$new_tests,na.rm=TRUE)
max_positive_rate_c=max(subset(figure_2_data,country==country_c)$positive_rate_smooth,na.rm=TRUE)
figure_2_c_3 <- (ggplot(subset(figure_2_data,country==country_c)) 
                 + geom_line(aes(x = date, y = new_tests),size=0.7,color=my_palette_1,na.rm=TRUE)
                 + geom_line(aes(x = date, y = new_tests_smooth),size=1,color=my_palette_2,na.rm=TRUE)
                 + geom_line(aes(x = date, y = positive_rate_smooth*(max_tests_c/max_positive_rate_c)),color=my_palette_3,na.rm=TRUE)
                 + scale_y_continuous(name = element_blank(), 
                                      expand = c(0,0),limits = c(0, NA),
                                      sec.axis = sec_axis(~./(max_tests_c/max_positive_rate_c), name = element_blank(),
                                                          breaks=c(0,0.05,0.10,0.15,0.20),
                                                          labels=c('0','0·05','0·10','0·15','0·20')))
                 + labs(x="Date")
                 + theme_light()
                 + theme(plot.title = element_text(hjust = 0.5), axis.line=element_line(color="black",size=0.7),axis.ticks=element_line(color="black",size=0.7),plot.margin=unit(c(0,0,0,0),"pt"),
                         axis.title.y.left = element_text(color=my_palette_2), axis.title.y.right = element_text(color=my_palette_3)))

figure_2_d_1 <- (ggplot(subset(figure_2_data,country==country_d)) 
                 + geom_line(aes(x = date, y = new_per_day),size=0.7,color=my_palette_1,na.rm=TRUE)
                 + geom_line(aes(x = date, y = new_per_day_smooth),size=1,color=my_palette_2,na.rm=TRUE)
                 + labs(title=country_d,x=element_blank(),y=element_blank())
                 + scale_y_continuous(expand = c(0,0),limits = c(0, NA))
                 + theme_light()
                 + theme(plot.title=element_text(size=12, hjust = 0.5), axis.line=element_line(color="black",size=0.7),axis.ticks=element_line(color="black",size=0.7),plot.margin=unit(c(0,0,0,0),"pt")))
figure_2_d_2 <- (ggplot(subset(figure_2_data,country==country_d))
                 + geom_line(aes(x = date, y = dead_per_day),size=0.7,color=my_palette_1,na.rm=TRUE)
                 + geom_line(aes(x = date, y = dead_per_day_smooth),size=1,color=my_palette_2,na.rm=TRUE)
                 + labs(x=element_blank(),y=element_blank())
                 + scale_y_continuous(expand = c(0,0),limits = c(0, NA))
                 + theme_light()
                 + theme(plot.title = element_text(hjust = 0.5), axis.line=element_line(color="black",size=0.7),axis.ticks=element_line(color="black",size=0.7),plot.margin=unit(c(0,0,0,0),"pt")))
max_tests_d=max(subset(figure_2_data,country==country_d)$new_tests,na.rm=TRUE)
max_positive_rate_d=max(subset(figure_2_data,country==country_d)$positive_rate_smooth,na.rm=TRUE)
figure_2_d_3 <- (ggplot(subset(figure_2_data,country==country_d)) 
                 + geom_line(aes(x = date, y = new_tests),size=0.7,color=my_palette_1,na.rm=TRUE)
                 + geom_line(aes(x = date, y = new_tests_smooth),size=1,color=my_palette_2,na.rm=TRUE)
                 + geom_line(aes(x = date, y = positive_rate_smooth*(max_tests_d/max_positive_rate_d)),color=my_palette_3,na.rm=TRUE)
                 + scale_y_continuous(name = element_blank(), 
                                      expand = c(0,0),limits = c(0, NA),
                                      sec.axis = sec_axis(~./(max_tests_d/max_positive_rate_d), name = "Positive Rate",
                                                          breaks=c(0,0.05,0.10,0.15,0.20,0.25,0.30),
                                                          labels=c('0','0·05','0·10','0·15','0·20','0·25','0·30')))
                 + labs(x="Date")
                 + theme_light()
                 + theme(plot.title = element_text(hjust = 0.5), axis.line=element_line(color="black",size=0.7),axis.ticks=element_line(color="black",size=0.7),plot.margin=unit(c(0,0,0,0),"pt"),
                         axis.title.y.left = element_text(color=my_palette_2), axis.title.y.right = element_text(color=my_palette_3)))


# Combining subplots
figure_2_all <- egg::ggarrange(figure_2_a_1,figure_2_b_1,figure_2_c_1,figure_2_d_1,
                               figure_2_a_2,figure_2_b_2,figure_2_c_2,figure_2_d_2,
                               figure_2_a_3,figure_2_b_3,figure_2_c_3,figure_2_d_3,
                               ncol=4)
figure_2_all <- annotate_figure(figure_2_all,
                                top = text_grob("Figure 2: Cases, Deaths and Testing Over Time", size = 14))

ggsave("./plots/figure_2.png", plot = figure_2_all, width = 16,  height = 8)
ggsave("./plots/figure_2.pdf", plot = figure_2_all, width = 16,  height = 8)

# Plot figure 4: USA Choropleth ---------------------------------------------------------------

# Import Data for figure 4 -------------------------------------------------------------------
# Import csv file for figure 4: Time series and choropleth for USA
figure_4a_data <- read_csv(file="./data/figure_4a.csv",
                           na = c("N/A","NA","#N/A"," ",""))
figure_4a_data$countrycode <- as.factor(figure_4a_data$countrycode)
figure_4a_data$adm_area_1 <- as.factor(figure_4a_data$adm_area_1)

figure_4b_data <- read_delim(file="./data/figure_4.csv",
                             delim=";",
                             na = c("N/A","NA","#N/A"," ","","None"))
figure_4b_data$gid <- as.factor(figure_4b_data$gid)
figure_4b_data$fips <- as.factor(figure_4b_data$fips)

# Process Data for figure 4 -------------------------------------------------------------------
# Figure 4a processing
# Get top n states by total confirmed cases, group others into Others
figure_4a_max <- aggregate(figure_4a_data[c("confirmed")],
                           by = list(figure_4a_data$adm_area_1),
                           FUN = max,
                           na.rm=TRUE)
figure_4a_max <- plyr::rename(figure_4a_max, c("Group.1"="adm_area_1"))
figure_4a_max <- figure_4a_max[order(-figure_4a_max$confirmed),]
n=15
top_n <- head(figure_4a_max$adm_area_1,n)
figure_4a_longitudes <- unique(figure_4a_data[figure_4a_data$adm_area_1%in%top_n,c("adm_area_1","longitude")])
figure_4a_longitudes <- figure_4a_longitudes[order(figure_4a_longitudes$longitude),]
figure_4a_longitudes <- figure_4a_longitudes$adm_area_1

figure_4a_data$State <- figure_4a_data$adm_area_1
levels(figure_4a_data$State) <- c(levels(figure_4a_data$State), "Others")
figure_4a_data[!figure_4a_data$adm_area_1%in%top_n,"State"] <- "Others"
figure_4a_data$State <- factor(figure_4a_data$State, levels=c(lapply(figure_4a_longitudes, as.character), "Others"))

figure_4a_agg <- aggregate(figure_4a_data[c("new_per_day_smooth")],
                           by = list(figure_4a_data$State,figure_4a_data$date),
                           FUN = sum,
                           na.rm=TRUE)
figure_4a_agg <- plyr::rename(figure_4a_agg, c("Group.1"="State","Group.2"="date"))

# Figure 4b processing
# Sort by GID and date
figure_4b_data <- figure_4b_data[order(figure_4b_data$gid, figure_4b_data$date),]
# Compute new cases per day as difference between daily case total
figure_4b_data[-1,"new_cases"] <- diff(figure_4b_data$cases)
# Remove first day of each GID as it does not have a value for new cases
figure_4b_data <- figure_4b_data[duplicated(figure_4b_data$gid),]
# Set any negative values for new cases to 0
figure_4b_data[figure_4b_data$new_cases<0,"new_cases"] <- 0
# Compute new cases per 10000 popuation
figure_4b_data$new_cases_per_10k <- 10000*figure_4b_data$new_cases/figure_4b_data$Population

# Define which dates to plot in choropleth
date_1 <- as.Date("2020-04-14")
date_2 <- as.Date("2020-07-21")
date_3 <- as.Date("2020-11-20")

# Subset for the two dates select
figure_4b1_data <- subset(figure_4b_data,date==date_1)
figure_4b2_data <- subset(figure_4b_data,date==date_2)
figure_4b3_data <- subset(figure_4b_data,date==date_3)

# Set max value to show. Censor any values above this 
color_max <- 250
figure_4b1_data$new_cases_censored <- figure_4b1_data$new_cases
figure_4b1_data$new_cases_censored[figure_4b1_data$new_cases_censored > color_max] <- color_max
figure_4b2_data$new_cases_censored <- figure_4b2_data$new_cases
figure_4b2_data$new_cases_censored[figure_4b2_data$new_cases_censored > color_max] <- color_max
figure_4b3_data$new_cases_censored <- figure_4b3_data$new_cases
figure_4b3_data$new_cases_censored[figure_4b3_data$new_cases_censored > color_max] <- color_max

# Convert the dataframe for figure_4b1 and 4b2 data into spatial dataframe
# Remove rows with NA in geometry. Required to convert column to shape object
figure_4b1_data <- subset(figure_4b1_data,!is.na(geometry))
figure_4b2_data <- subset(figure_4b2_data,!is.na(geometry))
figure_4b3_data <- subset(figure_4b3_data,!is.na(geometry))
# Convert "geometry" column to a sfc shape column 
figure_4b1_data$geometry <- st_as_sfc(figure_4b1_data$geometry)
figure_4b2_data$geometry <- st_as_sfc(figure_4b2_data$geometry)
figure_4b3_data$geometry <- st_as_sfc(figure_4b3_data$geometry)
# Convert dataframe to a sf shape object with "geometry" containing the shape information
figure_4b1_data <- st_sf(figure_4b1_data)
figure_4b2_data <- st_sf(figure_4b2_data)
figure_4b3_data <- st_sf(figure_4b3_data)


# Figure 4: USA time series and choropleth ----------------------------------------------
# Set up colour palette
my_palette_1 <- brewer.pal(name="YlGnBu",n=4)[2]
my_palette_2 <- brewer.pal(name="YlGnBu",n=4)[4]
my_palette_3 <- "GnBu"
my_palette_4 <- brewer.pal(name="Oranges",n=4)[4]
# Viridis color palette with last item gray
v_palette <-  viridis(n+1, option="D")
v_palette[n+1] <- "#C0C0C0"

# Figure 4a: Stacked Area Time series of US counties
figure_4a <-  (ggplot(data=figure_4a_agg, aes(x=date,y=new_per_day_smooth,fill=State))
               + geom_area(alpha=0.8, colour="white", na.rm=TRUE)
               + scale_fill_manual(values = v_palette)
               + labs(title="New Cases Over Time for US States", y="New Cases per Day (Smoothed)", x="Date")
               + scale_x_date(date_breaks="months", date_labels="%b")
               + scale_y_continuous(expand=c(0,0), limits=c(0, NA))
               + theme_light()
               + theme(plot.title = element_text(hjust = 0.5), axis.line=element_line(color="black",size=0.7),axis.ticks=element_line(color="black",size=0.7)
                       ,plot.margin=unit(c(0,0,0,0),"pt"), legend.position = c(0.09, 0.6),
                       legend.title = element_text(size = 12),legend.text = element_text(size = 11)))
figure_4a
ggsave("./plots/figure_4a.png", plot = figure_4a, width = 15,  height = 7)
ggsave("./plots/figure_4a.pdf", plot = figure_4a, width = 15,  height = 7)

# Figure 4b: Choropleth of US counties at USA peak dates
figure_4b1 <- (ggplot(data = figure_4b1_data) 
               + geom_sf(aes(fill=new_cases_censored), lwd=0, color=NA, na.rm=TRUE)
               + labs(title=paste("New Cases per Day per United States County at",date_1), fill="New Cases per Day")
               + scale_fill_distiller(palette=my_palette_3, trans="reverse", limits=c(color_max,0))
               + scale_x_continuous(expand=c(0,0), limits=c(-125, -65)) # coordinates are cropped to exclude Alaska
               + scale_y_continuous(expand=c(0,0), limits=c(24, 50))
               + theme_void()
               + guides(fill = guide_colourbar(barwidth = 30, barheight = 0.6, reverse=T))
               + theme(plot.title = element_text(hjust = 0.5), panel.grid.major=element_line(colour = "transparent"),legend.position="bottom"))
ggsave("./plots/figure_4b1.png", plot = figure_4b1, width = 9,  height = 7)
ggsave("./plots/figure_4b1.pdf", plot = figure_4b1, width = 9,  height = 7)

figure_4b2 <- (ggplot(data = figure_4b2_data) 
               + geom_sf(aes(fill=new_cases_censored), lwd=0, color=NA)
               + labs(title=paste("New Cases per Day per United States County at",date_2), fill="New Cases per Day")
               + scale_fill_distiller(palette=my_palette_3, trans="reverse", limits=c(color_max,0))
               + scale_x_continuous(expand=c(0,0), limits=c(-125, -65)) # coordinates are cropped to exclude Alaska
               + scale_y_continuous(expand=c(0,0), limits=c(24, 50))
               + theme_void()
               + guides(fill = guide_colourbar(barwidth = 30, barheight = 0.6, reverse=T))
               + theme(plot.title = element_text(hjust = 0.5), panel.grid.major=element_line(colour = "transparent"),legend.position="bottom"))
ggsave("./plots/figure_4b2.png", plot = figure_4b2, width = 9,  height = 7)
ggsave("./plots/figure_4b2.pdf", plot = figure_4b2, width = 9,  height = 7)

figure_4b3 <- (ggplot(data = figure_4b3_data) 
               + geom_sf(aes(fill=new_cases_censored), lwd=0, color=NA)
               + labs(title=paste("New Cases per Day per United States County at",date_3), fill="New Cases per Day")
               + scale_fill_distiller(palette=my_palette_3, trans="reverse", limits=c(color_max,0))
               + scale_x_continuous(expand=c(0,0), limits=c(-125, -65)) # coordinates are cropped to exclude Alaska
               + scale_y_continuous(expand=c(0,0), limits=c(24, 50))
               + theme_void()
               + guides(fill = guide_colourbar(barwidth = 30, barheight = 0.6, reverse=T))
               + theme(plot.title = element_text(hjust = 0.5), panel.grid.major=element_line(colour = "transparent"),legend.position="bottom"))
ggsave("./plots/figure_4b3.png", plot = figure_4b3, width = 9,  height = 7)
ggsave("./plots/figure_4b3.pdf", plot = figure_4b3, width = 9,  height = 7)



# Deprecated Code -----------------------------------------------------------------------


# figure_3b_wave <- (ggplot(figure_3b_wave_data, aes(x = si_at_t0_10_dead, y = dead_during_wave_per_10k, color=wave)) 
#                    + geom_point(size=1.5,alpha=0.9,stroke=1.5, na.rm=TRUE)
#                    + geom_text(data=subset(figure_3b_wave_data,
#                                            (countrycode %in% label_countries) |
#                                              (si_at_t0_10_dead <= quantile(figure_3b_wave_data$si_at_t0_10_dead, 0.1,na.rm=TRUE)) |
#                                              (si_at_t0_10_dead >= quantile(figure_3b_wave_data$si_at_t0_10_dead, 0.95,na.rm=TRUE))),
#                                aes(label=country),
#                                hjust=-0.1, vjust=-0.1,
#                                show.legend = FALSE)
#                    + theme_light()
#                    + theme(plot.title=element_text(hjust = 0.5), axis.line=element_line(color="black",size=0.7),axis.ticks=element_line(color="black",size=0.7), legend.position=c(0.6, 0.15))
#                    #+ scale_x_continuous(trans=pseudolog10_trans)
#                    + scale_y_continuous(trans='log10')
#                    + labs(title = "Total Deaths During Each Wave Against Stringency at the Date of 10th Death", x = "Stringency Index at the Date of 10th Death", y = "Total Deaths During Wave per 10,000 Population"))
# figure_3b_wave
# ggsave("./plots/figure_3b_wave_test_2.png", plot = figure_3b_wave, width = 9,  height = 7)
# 
# # Kendall's rank correlation test: p-value of 0.002569
# # Spearman's rank correlation test: p-value of 0.003223
# cor.test(figure_3b_data$SI_DAYS_TO_THRESHOLD, figure_3b_data$EPI_DEAD_PER_10K,
#          alternative = "greater",
#          method = "kendall")

# # Define list of mobility variables
# mobilities = c("workplace", "transit_stations", "retail_recreation", "residential")
# 
# # Calculate duration flags raised in first wave for Figure 3
# flags = c("SI",
#           "C1_SCHOOL_CLOSING",
#           "C2_WORKPLACE_CLOSING",
#           "C3_CANCEL_PUBLIC_EVENTS",
#           "C4_RESTRICTIONS_ON_GATHERINGS",
#           "C5_CLOSE_PUBLIC_TRANSPORT",
#           "C6_STAY_AT_HOME_REQUIREMENTS",
#           "C7_RESTRICTIONS_ON_INTERNAL_MOVEMENT",
#           "C8_INTERNATIONAL_TRAVEL_CONTROLS",
#           "H2_TESTING_POLICY",
#           "H3_CONTACT_TRACING")
# 
# for (flag in flags){
#   figure_3b_data[,paste(flag,"_DAYS_ABOVE_THRESHOLD_FIRST_WAVE_PROPORTION",sep="")] <- figure_3b_data[,paste(flag,"_DAYS_ABOVE_THRESHOLD_FIRST_WAVE",sep="")]/figure_3b_data[,"EPI_DURATION_FIRST_WAVE"]
# }

# # Figure 3: Testing other variables for scatter plot ----------------------------------------
# # Create directories. y variables: duration of first wave, or cumulative deaths per 10k.
# dir.create('./plots/figure_3b/y_duration/x_days_to_threshold')
# dir.create('./plots/figure_3b/y_dead/x_days_to_threshold')
# dir.create('./plots/figure_3b/y_duration/x_proportion_first_wave')
# dir.create('./plots/figure_3b/y_dead/x_proportion_first_wave')
# 
# # Define explanatory variables
# x_vars = c("GOV_MAX_SI_DAYS_FROM_T0",
#            "MAX_SI",
#            "SI_AT_PEAK_1",
#            "SI_DAYS_ABOVE_THRESHOLD_FIRST_WAVE_PROPORTION",
#            "C1_SCHOOL_CLOSING_DAYS_ABOVE_THRESHOLD_FIRST_WAVE_PROPORTION",
#            "C2_WORKPLACE_CLOSING_DAYS_ABOVE_THRESHOLD_FIRST_WAVE_PROPORTION",
#            "C3_CANCEL_PUBLIC_EVENTS_DAYS_ABOVE_THRESHOLD_FIRST_WAVE_PROPORTION",
#            "C4_RESTRICTIONS_ON_GATHERINGS_DAYS_ABOVE_THRESHOLD_FIRST_WAVE_PROPORTION",
#            "C5_CLOSE_PUBLIC_TRANSPORT_DAYS_ABOVE_THRESHOLD_FIRST_WAVE_PROPORTION",
#            "C6_STAY_AT_HOME_REQUIREMENTS_DAYS_ABOVE_THRESHOLD_FIRST_WAVE_PROPORTION",
#            "C7_RESTRICTIONS_ON_INTERNAL_MOVEMENT_DAYS_ABOVE_THRESHOLD_FIRST_WAVE_PROPORTION",
#            "C8_INTERNATIONAL_TRAVEL_CONTROLS_DAYS_ABOVE_THRESHOLD_FIRST_WAVE_PROPORTION",
#            "H2_TESTING_POLICY_DAYS_ABOVE_THRESHOLD_FIRST_WAVE_PROPORTION",
#            "H3_CONTACT_TRACING_DAYS_ABOVE_THRESHOLD_FIRST_WAVE_PROPORTION",
#            "SI_DAYS_TO_THRESHOLD",
#            "C1_SCHOOL_CLOSING_DAYS_TO_THRESHOLD",
#            "C2_WORKPLACE_CLOSING_DAYS_TO_THRESHOLD",
#            "C3_CANCEL_PUBLIC_EVENTS_DAYS_TO_THRESHOLD",
#            "C4_RESTRICTIONS_ON_GATHERINGS_DAYS_TO_THRESHOLD",
#            "C5_CLOSE_PUBLIC_TRANSPORT_DAYS_TO_THRESHOLD",
#            "C6_STAY_AT_HOME_REQUIREMENTS_DAYS_TO_THRESHOLD",
#            "C7_RESTRICTIONS_ON_INTERNAL_MOVEMENT_DAYS_TO_THRESHOLD",
#            "C8_INTERNATIONAL_TRAVEL_CONTROLS_DAYS_TO_THRESHOLD",
#            "H2_TESTING_POLICY_DAYS_TO_THRESHOLD",
#            "H3_CONTACT_TRACING_DAYS_TO_THRESHOLD")
# 
# for (x in x_vars) {
#   y = "EPI_DEAD_PER_10K" # with log axis
#   figure_3b_loop <- (ggplot(figure_3b_data, aes_string(x = x, y = y, colour = "CLASS")) 
#                      + geom_point(size=1.5,shape=1,alpha=0.9,stroke=1.5, na.rm=TRUE)
#                      + geom_text(data=subset(figure_3b_data,
#                                              (COUNTRYCODE %in% label_countries)),
#                                  aes(label=COUNTRY),
#                                  hjust=-0.1, vjust=-0.1,
#                                  show.legend = FALSE)
#                      + theme_light()
#                      + scale_y_continuous(trans='log10', breaks = log_breaks(n=10,base=10))
#                      + theme(plot.title=element_text(hjust = 0.5), axis.line=element_line(color="black",size=0.7),axis.ticks=element_line(color="black",size=0.7))
#                      + scale_color_manual(values = my_palette_1, name = "Epidemic Wave State", labels = c("Entering First Wave", "Past First Wave", "Entering Second Wave","Past Second Wave")))
#   if (grepl('DAYS_TO_THRESHOLD', x, fixed=TRUE)){
#     ggsave(paste("./plots/figure_3b/y_dead/x_days_to_threshold/figure_3b_",x,"_",y,".png",sep=''), plot = figure_3b_loop, width = 9,  height = 7)
#   } else if (grepl('DAYS_ABOVE_THRESHOLD_FIRST_WAVE_PROPORTION', x, fixed=TRUE)){
#     ggsave(paste("./plots/figure_3b/y_dead/x_proportion_first_wave/figure_3b_",x,"_",y,".png",sep=''), plot = figure_3b_loop, width = 9,  height = 7)
#   } else {
#     ggsave(paste("./plots/figure_3b/y_dead/figure_3b_",x,"_",y,".png",sep=''), plot = figure_3b_loop, width = 9,  height = 7)
#   }
#   y = "EPI_DURATION_FIRST_WAVE"  # with linear axis
#   figure_3b_loop <- (ggplot(figure_3b_data, aes_string(x = x, y = y, colour = "CLASS")) 
#                      + geom_point(size=1.5,shape=1,alpha=0.9,stroke=1.5, na.rm=TRUE)
#                      + geom_text(data=subset(figure_3b_data,
#                                              (COUNTRYCODE %in% label_countries)),
#                                  aes(label=COUNTRY),
#                                  hjust=-0.1, vjust=-0.1,
#                                  show.legend = FALSE)
#                      + theme_light()
#                      + theme(plot.title=element_text(hjust = 0.5), axis.line=element_line(color="black",size=0.7),axis.ticks=element_line(color="black",size=0.7))
#                      + scale_color_manual(values = my_palette_1, name = "Epidemic Wave State", labels = c("Entering First Wave", "Past First Wave", "Entering Second Wave","Past Second Wave")))
#   if (grepl('DAYS_TO_THRESHOLD', x, fixed=TRUE)){
#     ggsave(paste("./plots/figure_3b/y_duration/x_days_to_threshold/figure_3b_",x,"_",y,".png",sep=''), plot = figure_3b_loop, width = 9,  height = 7)
#   } else if (grepl('DAYS_ABOVE_THRESHOLD_FIRST_WAVE_PROPORTION', x, fixed=TRUE)){
#     ggsave(paste("./plots/figure_3b/y_duration/x_proportion_first_wave/figure_3b_",x,"_",y,".png",sep=''), plot = figure_3b_loop, width = 9,  height = 7)
#   } else {
#     ggsave(paste("./plots/figure_3b/y_duration/figure_3b_",x,"_",y,".png",sep=''), plot = figure_3b_loop, width = 9,  height = 7)
#   }
# }

# # Import csv file for Figure 3a
# figure_3a_data <- readr::read_csv("./data/figure_3a.csv", 
#                                   na = c("N/A","NA","#N/A"," ",""))
# figure_3a_data$COUNTRYCODE = as.factor(figure_3a_data$COUNTRYCODE)
# figure_3a_data$COUNTRY = as.factor(figure_3a_data$COUNTRY)
# figure_3a_data$CLASS = as.factor(figure_3a_data$CLASS)
# 
# # Import csv file for Figure 3b
# figure_3b_data <- read_csv("./data/figure_3b.csv", 
#                            na = c("N/A","NA","#N/A"," ",""))
# figure_3b_data$COUNTRYCODE = as.factor(figure_3b_data$COUNTRYCODE)
# figure_3b_data$COUNTRY = as.factor(figure_3b_data$COUNTRY)
# figure_3b_data$CLASS = as.factor(figure_3b_data$CLASS)

# figure_3a_data <- subset(figure_3a_data,CLASS%in%c(1,2,3,4))
# figure_3b_data <- subset(figure_3b_data,CLASS%in%c(1,2,3,4))
# figure_3b_data <- subset(figure_3b_data,COUNTRY!='Guinea-Bissau')
# 
# # Aggregate data by class and t_1_dead, mean mean and sd for each date
# # Figure 3 mean and standard errors
# figure_3a_agg <- aggregate(figure_3a_data[c("stringency_index")],
#                            by = list(figure_3a_data$CLASS, figure_3a_data$t_1_dead),
#                            FUN = mean,
#                            na.action = na.pass)
# figure_3a_agg <- plyr::rename(figure_3a_agg, c("Group.1"="CLASS", "Group.2"="t_1_dead","stringency_index"="mean_si"))
# figure_3a_se <- aggregate(figure_3a_data[c("stringency_index")],
#                           by = list(figure_3a_data$CLASS, figure_3a_data$t_1_dead),
#                           FUN = std.error)
# figure_3a_se <- plyr::rename(figure_3a_se, c("Group.1"="CLASS", "Group.2"="t_1_dead","stringency_index"="se_si"))
# figure_3a_agg <- merge(figure_3a_agg,figure_3a_se, by=c("CLASS","t_1_dead"))
# 
# 
# # Figure 3 mean and standard errors
# figure_3b_agg <- aggregate(figure_3b_data[mobilities],
#                            by = list(figure_3b_data$CLASS, figure_3b_data$t_1_dead),
#                            FUN = mean)
# figure_3b_agg <- plyr::rename(figure_3b_agg, c("Group.1"="CLASS", "Group.2"="t_1_dead"))
# figure_3b_se <- aggregate(figure_3b_data[mobilities],
#                           by = list(figure_3b_data$CLASS, figure_3b_data$t_1_dead),
#                           FUN = std.error)
# figure_3b_se <- plyr::rename(figure_3b_se, c("Group.1"="CLASS", "Group.2"="t_1_dead"))
# figure_3b_agg <- merge(figure_3b_agg,figure_3b_se, by=c("CLASS","t_1_dead"))
# 
# 
# # Get the number of elements in each class to work out the t_1_dead xlim values
# figure_3a_count <- aggregate(figure_3a_data[c("stringency_index")],
#                              by = list(figure_3a_data$CLASS, figure_3a_data$t_1_dead),
#                              FUN = length)
# figure_3a_count <- plyr::rename(figure_3a_count, c("Group.1"="CLASS", "Group.2"="t_1_dead","stringency_index"="n_present"))
# figure_3a_count_max <- aggregate(figure_3a_count[c("n_present")],
#                                  by = list(figure_3a_count$CLASS),
#                                  FUN = max)
# figure_3a_count_max <- plyr::rename(figure_3a_count_max, c("Group.1"="CLASS","n_present"="n_total"))
# figure_3a_count <- merge(figure_3a_count,figure_3a_count_max, by="CLASS")
# 
# figure_3b_count <- aggregate(figure_3b_data[c("residential")],
#                              by = list(figure_3b_data$CLASS, figure_3b_data$t_1_dead),
#                              FUN = length)
# figure_3b_count <- plyr::rename(figure_3b_count, c("Group.1"="CLASS", "Group.2"="t_1_dead","residential"="n_present"))
# figure_3b_count_max <- aggregate(figure_3b_count[c("n_present")],
#                                  by = list(figure_3b_count$CLASS),
#                                  FUN = max)
# figure_3b_count_max <- plyr::rename(figure_3b_count_max, c("Group.1"="CLASS","n_present"="n_total"))
# figure_3b_count <- merge(figure_3b_count,figure_3b_count_max, by="CLASS")
# 
# # n_threshold determines where to cut off t_1_dead xlim values. Only takes t_1_dead values for which there are >= n_threshold % of the total present for each class
# n_threshold = 0.8
# figure_3a_count <- subset(figure_3a_count, n_present>=n_threshold*n_total)
# figure_3b_count <- subset(figure_3b_count, n_present>=n_threshold*n_total)
# 
# t_min = min(figure_3a_count$t_1_dead, figure_3b_count$t_1_dead)
# t_max = max(figure_3a_count$t_1_dead, figure_3b_count$t_1_dead)

# Figure 3: Scatter plot of government response time against number of cases for each country
# figure_3b <- (ggplot(figure_3b_data, aes(x = SI_DAYS_TO_THRESHOLD, y = EPI_DEAD_PER_10K, colour = CLASS)) 
#               + geom_point(size=1.5,shape=1,alpha=0.9,stroke=1.5, na.rm=TRUE)
#               #+ geom_vline(xintercept=0,linetype="dashed", color=my_palette_2, size=1)
#               #+ annotate("text",x=2,y=490,hjust=0,label="T0",color=my_palette_2)
#               # Label countries that have high number of deaths, or early/late government response times
#               + geom_text(data=subset(figure_3b_data,
#                                       (COUNTRYCODE %in% label_countries) |
#                                         #(EPI_DEAD_PER_10K >= quantile(figure_3b_data$EPI_DEAD_PER_10K, 0.95,na.rm=TRUE)) |
#                                         (SI_DAYS_TO_THRESHOLD >= quantile(figure_3b_data$SI_DAYS_TO_THRESHOLD, 0.95,na.rm=TRUE)) |
#                                         (SI_DAYS_TO_THRESHOLD <= quantile(figure_3b_data$SI_DAYS_TO_THRESHOLD, 0.05,na.rm=TRUE))),
#                           aes(label=COUNTRY),
#                           hjust=-0.1, vjust=-0.1,
#                           show.legend = FALSE)
#               + theme_light()
#               + theme(plot.title=element_text(hjust = 0.5), axis.line=element_line(color="black",size=0.7),axis.ticks=element_line(color="black",size=0.7), legend.position=c(0.6, 0.15))
#               + scale_color_manual(values = my_palette_1, name = "Epidemic Wave State", labels = c("Entering First Wave", "Past First Wave", "Entering Second Wave","Past Second Wave"))
#               + scale_x_continuous(trans=pseudolog10_trans, breaks=c(-200,-100,-30,-10,-3,-1,0,1,3,10,30,100,200), expand=expand_scale(mult=c(0.05,0.2)))
#               + scale_y_continuous(trans='log10', breaks=c(0.001,0.01,0.1,1,10), labels=c(0.001,0.01,0.1,1,10))
#               + labs(title = "Total Deaths Against Government Response Time", x = "Government Response Time (Days from First Death to Stringency Index of 59 or Above)", y = "Total Deaths per 10,000 Population"))
# ggsave("./plots/figure_3b.png", plot = figure_3b, width = 9,  height = 7)

# # # Figure 3: Line plot of stringency index over time for each country class
# # figure_3a_loess <- (ggplot(figure_3a_data, aes(x = t_1_dead, y = stringency_index, colour = CLASS)) 
# #               #+ geom_line(aes(group=interaction(CLASS,COUNTRY),color=CLASS), size=0.1, alpha = 0.3,na.rm=TRUE)
# #               + geom_smooth(method="loess", level=0.95, span=0.3, na.rm=TRUE)
# #               + geom_vline(xintercept=0,linetype="dashed", color=my_palette_2, size=1)
# #               + annotate("text",x=2,y=97,hjust=0,label="T0 (First Day Surpassing Cumulative 5 Cases per Million)",color=my_palette_2)
# #               + theme_light()
# #               + coord_cartesian(xlim=c(t_min, t_max))
# #               + scale_color_manual(values = my_palette_1, name = "Epidemic Wave State", labels = c("Entering First Wave", "Past First Wave", "Entering Second Wave","Past Second Wave"))
# #               + theme(plot.title=element_text(hjust = 0.5), axis.line=element_line(color="black",size=0.7),axis.ticks=element_line(color="black",size=0.7))
# #               + scale_x_continuous(breaks=seq(floor(t_min/10)*10,ceiling(t_max/10)*10,10),expand=c(0,0),limits=c(t_min,t_max))
# #               + scale_y_continuous(breaks=seq(0,100,10),expand = c(0,0),limits = c(0, 100))
# #               + labs(title = "Average Stringency Index Over Time", x = "Days Since T0", y = "Stringency Index"))
# # ggsave("./plots/figure_3a_loess.png", plot = figure_3a_loess, width = 9,  height = 7)
# 
# figure_3a <- (ggplot(figure_3a_agg, aes(x = t_1_dead, y = mean_si, colour = CLASS)) 
#               + geom_line(size=1,show.legend = FALSE,na.rm=TRUE)
#               + geom_ribbon(aes(ymin=mean_si-se_si, ymax=mean_si+se_si, fill = CLASS), linetype=2, alpha=0.1, show.legend = FALSE)
#               #+ geom_vline(xintercept=0,linetype="dashed", color=my_palette_2, size=1)
#               #+ annotate("text",x=2,y=97,hjust=0,label="T0 (First Day Surpassing Cumulative 5 Cases per Million)",color=my_palette_2)
#               + theme_light()
#               + coord_cartesian(xlim=c(t_min, t_max))
#               + scale_color_manual(values = my_palette_1, name = "Epidemic Wave State", labels = c("Entering First Wave", "Past First Wave", "Entering Second Wave","Past Second Wave"))
#               + theme(plot.title=element_text(hjust = 0.5), axis.line=element_line(color="black",size=0.7),axis.ticks=element_line(color="black",size=0.7))
#               + scale_x_continuous(breaks=seq(floor(t_min/10)*10,ceiling(t_max/10)*10,10),expand=c(0,0),limits=c(t_min,t_max))
#               + scale_y_continuous(breaks=seq(0,100,10),expand = c(0,0),limits = c(0, 100))
#               + labs(title = "Average Stringency Index Over Time", x = "Days Since First Recorded Death", y = "Stringency Index"))
# ggsave("./plots/figure_3a.png", plot = figure_3a, width = 9,  height = 7)
# 
# # Figure 3: Line plot of residential mobility over time for each country class
# # Figure 3 for each mobility with loess smoothing
# # for (mobility in mobilities)
# # {
# #   figure_3b_loess <- (ggplot(figure_3b_data, aes_string(x = "t_1_dead", y = mobility, colour = "CLASS")) 
# #                       + geom_smooth(method="loess", level=0.95, span=0.3, na.rm=TRUE, show.legend=FALSE)
# #                       + geom_vline(xintercept=0,linetype="dashed", color=my_palette_2, size=1)
# #                       #+ annotate("text",x=2,y=27,hjust=0,label="T0 (First Day Surpassing Cumulative 5 Cases per Million)",color=my_palette_2)
# #                       + theme_light()
# #                       + coord_cartesian(xlim=c(t_min, t_max))
# #                       + scale_color_manual(values = my_palette_1, name = "Epidemic Wave State", labels = c("Entering First Wave", "Past First Wave", "Entering Second Wave","Past Second Wave"))
# #                       + theme(plot.title=element_text(hjust = 0.5), axis.line=element_line(color="black",size=0.7),axis.ticks=element_line(color="black",size=0.7))
# #                       + scale_x_continuous(breaks=seq(floor(t_min/10)*10,ceiling(t_max/10)*10,10),expand=c(0,0),limits=c(t_min,t_max))
# #                       + scale_y_continuous(expand = c(0,0))
# #                       + labs(title = paste("Average ",mobility," Mobility Over Time",sep=""), x = "Days Since T0", y = paste(mobility," Mobility (Change from Baseline, Smoothed)",sep="")))
# #   ggsave(paste("./plots/figure_3b_loess_",mobility,".png",sep=''), plot = figure_3b_loess, width = 9,  height = 7)
# # }
# 
# # Figure 3 for each mobility with mean and sd
# for (mobility in mobilities)
# {
#   figure_3b <- (ggplot(figure_3b_agg, aes_string(x = "t_1_dead", y = paste(mobility,".x",sep=""), colour = "CLASS")) 
#                 + geom_line(size=1,show.legend = FALSE,na.rm=TRUE)
#                 + geom_ribbon(aes_string(ymin=paste(mobility,".x-",mobility,".y",sep=""), ymax=paste(mobility,".x+",mobility,".y",sep=""), fill = "CLASS"), linetype=2, alpha=0.1, show.legend = FALSE)
#                 #+ geom_vline(xintercept=0,linetype="dashed", color=my_palette_2, size=1)
#                 #+ annotate("text",x=2,y=27,hjust=0,label="T0 (First Day Surpassing Cumulative 5 Cases per Million)",color=my_palette_2)
#                 + theme_light()
#                 + coord_cartesian(xlim=c(t_min, t_max))
#                 + scale_color_manual(values = my_palette_1, name = "Epidemic Wave State", labels = c("Entering First Wave", "Past First Wave", "Entering Second Wave","Past Second Wave"))
#                 + theme(plot.title=element_text(hjust = 0.5), axis.line=element_line(color="black",size=0.7),axis.ticks=element_line(color="black",size=0.7))
#                 + scale_x_continuous(breaks=seq(floor(t_min/10)*10,ceiling(t_max/10)*10,10),expand=c(0,0),limits=c(t_min,t_max))
#                 + scale_y_continuous(expand = c(0,0))
#                 + labs(title = paste("Average ",mobility," Mobility Over Time",sep=""), x = "Days Since First Recorded Death", y = paste(mobility," Mobility (Change from Baseline, Smoothed)",sep="")))
#   ggsave(paste("./plots/figure_3b_",mobility,".png",sep=''), plot = figure_3b, width = 9,  height = 7)
# }


# # Robustness check for the sensitivity of T0 to threshold: try generating figure_3 with T0 at 1, 5, 10 deaths
# figure_3a_agg <- aggregate(figure_3a_data[c("stringency_index")],
#                            by = list(figure_3a_data$CLASS, figure_3a_data$t_10_dead),
#                            FUN = mean,
#                            na.action = na.pass)
# figure_3a_agg <- plyr::rename(figure_3a_agg, c("Group.1"="CLASS", "Group.2"="t_10_dead","stringency_index"="mean_si"))
# figure_3a_se <- aggregate(figure_3a_data[c("stringency_index")],
#                           by = list(figure_3a_data$CLASS, figure_3a_data$t_10_dead),
#                           FUN = std.error)
# figure_3a_se <- plyr::rename(figure_3a_se, c("Group.1"="CLASS", "Group.2"="t_10_dead","stringency_index"="se_si"))
# figure_3a_agg <- merge(figure_3a_agg,figure_3a_se, by=c("CLASS","t_10_dead"))
# 
# figure_3a_count <- aggregate(figure_3a_data[c("stringency_index")],
#                              by = list(figure_3a_data$CLASS, figure_3a_data$t_10_dead),
#                              FUN = length)
# figure_3a_count <- plyr::rename(figure_3a_count, c("Group.1"="CLASS", "Group.2"="t_10_dead","stringency_index"="n_present"))
# figure_3a_count_max <- aggregate(figure_3a_count[c("n_present")],
#                                  by = list(figure_3a_count$CLASS),
#                                  FUN = max)
# figure_3a_count_max <- plyr::rename(figure_3a_count_max, c("Group.1"="CLASS","n_present"="n_total"))
# figure_3a_count <- merge(figure_3a_count,figure_3a_count_max, by="CLASS")
# n_threshold = 0.75
# figure_3a_count <- subset(figure_3a_count, n_present>=n_threshold*n_total)
# t_min = min(figure_3a_count$t_10_dead)
# t_max = max(figure_3a_count$t_10_dead)
# 
# figure_3a <- (ggplot(figure_3a_agg, aes(x = t_10_dead, y = mean_si, colour = CLASS)) 
#               + geom_line(size=1,na.rm=TRUE)
#               + geom_ribbon(aes(ymin=mean_si-se_si, ymax=mean_si+se_si, fill = CLASS), linetype=2, alpha=0.1, show.legend = FALSE)
#               + theme_light()
#               + scale_color_manual(values = my_palette_1, name = "Epidemic Wave State", labels = c("Entering First Wave", "Past First Wave", "Entering Second Wave","Past Second Wave"))
#               + theme(plot.title=element_text(hjust = 0.5), axis.line=element_line(color="black",size=0.7),axis.ticks=element_line(color="black",size=0.7))
#               + scale_x_continuous(breaks=seq(floor(t_min/10)*10,ceiling(t_max/10)*10,10),expand=c(0,0),limits=c(t_min,t_max))
#               + scale_y_continuous(breaks=seq(0,100,10),expand = c(0,0),limits = c(0, 100))
#               + labs(title = "Average Stringency Index Over Time", x = "Days Since 10th Recorded Death", y = "Stringency Index"))
# ggsave("./plots/figure_3a_t_10_dead.png", plot = figure_3a, width = 12,  height = 7)
# 
# 
# figure_3c_data$SI_DAYS_TO_THRESHOLD_5_DEAD <- figure_3c_data$SI_DAYS_TO_THRESHOLD - (figure_3c_data$T0_5_DEAD - figure_3c_data$T0_1_DEAD)
# figure_3c_data$SI_DAYS_TO_THRESHOLD_10_DEAD <- figure_3c_data$SI_DAYS_TO_THRESHOLD - (figure_3c_data$T0_10_DEAD - figure_3c_data$T0_1_DEAD)
# 
# figure_3c <- (ggplot(figure_3c_data, aes(x = SI_DAYS_TO_THRESHOLD_10_DEAD, y = EPI_DEAD_PER_10K, colour = CLASS)) 
#               + geom_point(size=1.5,shape=1,alpha=0.9,stroke=1.5, na.rm=TRUE)              
#               + geom_text(data=subset(figure_3c_data,
#                                       (COUNTRYCODE %in% label_countries) |
#                                         (SI_DAYS_TO_THRESHOLD >= quantile(figure_3c_data$SI_DAYS_TO_THRESHOLD, 0.95,na.rm=TRUE)) |
#                                         (SI_DAYS_TO_THRESHOLD <= quantile(figure_3c_data$SI_DAYS_TO_THRESHOLD, 0.05,na.rm=TRUE))),
#                           aes(label=COUNTRY),
#                           hjust=-0.1, vjust=-0.1,
#                           show.legend = FALSE)
#               + theme_light()
#               + theme(plot.title=element_text(hjust = 0.5), axis.line=element_line(color="black",size=0.7),axis.ticks=element_line(color="black",size=0.7), legend.position=c(0.8, 0.2))
#               + scale_color_manual(values = my_palette_1, name = "Epidemic Wave State", labels = c("Entering First Wave", "Past First Wave", "Entering Second Wave","Past Second Wave"))
#               + scale_x_continuous(breaks=seq(-175,175,25),expand = c(0,0),limits = c(-175, 175))
#               + scale_y_continuous(trans='log10', breaks = log_breaks(n=10,base=10))
#               + labs(title = "Total Deaths Against Government Response Time", x = "Government Response Time (Days from 10th Death to Stringency Index of 60 or Above)", y = "Total Deaths per 10,000 Population"))
# ggsave("./plots/figure_3c_t_10_dead.png", plot = figure_3c, width = 9,  height = 7)

# # Figure 5: Scatter plot of 1st wave vs 2nd wave magnitude ----------------------------------------------------------
# # Import data for figure 5 (same csv as figure_3c)
# figure_5_data <- read_csv("./data/figure_3c.csv", 
#                           na = c("N/A","NA","#N/A"," ",""))
# figure_5_data$COUNTRYCODE = as.factor(figure_5_data$COUNTRYCODE)
# figure_5_data$COUNTRY = as.factor(figure_5_data$COUNTRY)
# figure_5_data$CLASS = as.factor(figure_5_data$CLASS)
# figure_5_data$CLASS_COARSE = as.factor(figure_5_data$CLASS_COARSE)
# 
# # Normalize by population
# figure_5_data$DEAD_FIRST_WAVE_PER_10K <- 10000 * figure_5_data$DEAD_FIRST_WAVE / figure_5_data$POPULATION
# figure_5_data$DEAD_SECOND_WAVE_PER_10K <- 10000 * figure_5_data$DEAD_SECOND_WAVE / figure_5_data$POPULATION
# figure_5_data$DEAD_PEAK_1_PER_10K <- 10000 * figure_5_data$DEAD_PEAK_1 / figure_5_data$POPULATION
# figure_5_data$DEAD_PEAK_2_PER_10K <- 10000 * figure_5_data$DEAD_PEAK_2 / figure_5_data$POPULATION

# # Plot figure 5: scatterplot of magnitude of 1st wave against magnitude of second wave ----------------
# figure_5_a <- (ggplot(figure_5_data, aes(x = DEAD_FIRST_WAVE_PER_10K, y = DEAD_SECOND_WAVE_PER_10K)) 
#                + geom_point(size=1.5,shape=1,alpha=0.9,stroke=1.5, na.rm=TRUE)
#                + geom_text(data=figure_5_data,
#                            aes(label=COUNTRY),
#                            hjust=-0.1, vjust=-0.1,
#                            show.legend = FALSE)
#                + theme_light()
#                + theme(plot.title=element_text(hjust = 0.5), axis.line=element_line(color="black",size=0.7),axis.ticks=element_line(color="black",size=0.7), legend.position=c(0.8, 0.2))
#                #+ scale_color_manual(values = my_palette_1, name = "Epidemic Wave State", labels = c("Entering First Wave", "Past First Wave", "Entering Second Wave","Past Second Wave"))
#                + scale_x_continuous(trans='log10', breaks = log_breaks(n=10,base=10))
#                + scale_y_continuous(trans='log10', breaks = log_breaks(n=10,base=10))
#                + labs(title = "Total Deaths in First Wave Against Second Wave", x = "Total Number of Deaths in First Wave per 10,000 Population", y = "Total Number of Deaths in Second Wave per 10,000 Population"))
# ggsave("./plots/figure_5_a.png", plot = figure_5_a, width = 7,  height = 7)
# 
# figure_5_b <- (ggplot(figure_5_data, aes(x = DEAD_PEAK_1_PER_10K, y = DEAD_PEAK_2_PER_10K)) 
#                + geom_point(size=1.5,shape=1,alpha=0.9,stroke=1.5, na.rm=TRUE)
#                + geom_text(data=figure_5_data,
#                            aes(label=COUNTRY),
#                            hjust=-0.1, vjust=-0.1,
#                            show.legend = FALSE)
#                + theme_light()
#                + theme(plot.title=element_text(hjust = 0.5), axis.line=element_line(color="black",size=0.7),axis.ticks=element_line(color="black",size=0.7), legend.position=c(0.8, 0.2))
#                #+ scale_color_manual(values = my_palette_1, name = "Epidemic Wave State", labels = c("Entering First Wave", "Past First Wave", "Entering Second Wave","Past Second Wave"))
#                + scale_x_continuous(trans='log10', breaks = log_breaks(n=10,base=10))
#                + scale_y_continuous(trans='log10', breaks = log_breaks(n=10,base=10))
#                + labs(title = "New Deaths per Day in First Wave Against Second Wave", x = "New Deaths per Day at First Peak per 10,000 Population", y = "New Deaths per Day at Second Peak per 10,000 Population"))
# ggsave("./plots/figure_5_b.png", plot = figure_5_b, width = 7,  height = 7)

# # Melt dataframe to long form - non-smoothed
# figure_2_data_plot <- melt(subset(figure_2_data,country%in%c(country_a,country_b,country_c)),
#                            id.vars=c("country","countrycode","date"),
#                            measure.vars=c("new_per_day","dead_per_day","new_tests"),
#                            na.rm=TRUE)
# # Melt dataframe to long form - smoothed
# figure_2_data_plot_smooth <- melt(subset(figure_2_data,country%in%c(country_a,country_b,country_c)),
#                                   id.vars=c("country","countrycode","date"),
#                                   measure.vars=c("new_per_day_smooth","dead_per_day_smooth","new_tests_smoothed"),
#                                   na.rm=TRUE)
# # Melt dataframe to long form - positive rate
# figure_2_data_plot_positive <- melt(subset(figure_2_data,country%in%c(country_a,country_b,country_c)),
#                                   id.vars=c("country","countrycode","date"),
#                                   measure.vars=c("positive_rate"),
#                                   na.rm=TRUE)
# # rename values
# figure_2_data_plot <- figure_2_data_plot %>% mutate(variable=recode(variable,
#                                                                     "new_per_day"="New Cases per Day",
#                                                                     "dead_per_day"="Deaths per Day",
#                                                                     "new_tests"="Tests per Day"))
# figure_2_data_plot_smooth <- figure_2_data_plot_smooth %>% mutate(variable=recode(variable,
#                                                                                   "new_per_day_smooth"="New Cases per Day",
#                                                                                   "dead_per_day_smooth"="Deaths per Day",
#                                                                                   "new_tests_smoothed"="Tests per Day"))
# 
# # Plot Figure 2 - with facet grid ------------------------------------------------------------
# # Set up colour palette
# my_palette_1 <- brewer.pal(name="YlGnBu",n=4)[2]
# my_palette_2 <- brewer.pal(name="YlGnBu",n=4)[4]
# my_palette_3 <- brewer.pal(name="Oranges",n=4)[4]
# 
# 
# figure_2_grid <- (ggplot()
#                   + geom_line(data=figure_2_data_plot, aes(x = date, y = value),size=1,color=my_palette_1,na.rm=TRUE)
#                   + geom_line(data=figure_2_data_plot_smooth, aes(x = date, y = value),size=1,color=my_palette_2,na.rm=TRUE)
#                   + facet_wrap(variable~country, scales="free", nrow=3)
#                   + theme_light()
#                   + scale_y_continuous(expand = c(0,0),limits = c(0, NA))
#                   + theme(plot.title = element_text(hjust = 0.5), axis.line=element_line(color="black",size=0.7),axis.ticks=element_line(color="black",size=0.7),plot.margin=unit(c(0,0,0,0),"pt")))
# # Need to:
# # Add positive rate
# # Add y axis labels for each variable
# # Title and axis labels
# 
# ggsave("./plots/figure_2_grid.png", plot = figure_2_grid, width = 12,  height = 9)

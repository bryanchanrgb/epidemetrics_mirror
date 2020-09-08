# Epidemetrics - Generate Plots (Figures 2, 3, 4, 5)

# Load Packages, Clear, Sink -------------------------------------------------------

# load packages
package_list <- c("readr","ggplot2","gridExtra","plyr","dplyr","viridis","stats","ggsci","RColorBrewer")
for (package in package_list){
  if (!package %in% installed.packages()){
    install.packages(package)
  }
}
lapply(package_list, require, character.only = TRUE)

# clear workspace
rm(list=ls())

# Import Data -------------------------------------------------------------

# Import csv file for Figure 2
figure_2_data <- read_csv("Data/figure_2.csv", 
                          na = c("N/A","NA","#N/A"," ",""),
                          col_types = cols(COUNTRYCODE = col_factor(levels = NULL),
                                           COUNTRY = col_factor(levels = NULL),
                                           CLASS = col_factor(levels = c(1,2,3,4,0))))

# Import csv file for figure 3
figure_3_data <- read_csv("Data/figure_3.csv", 
                          na = c("N/A","NA","#N/A"," ",""),
                          col_types = cols(COUNTRYCODE = col_factor(levels = NULL),
                                           COUNTRY = col_factor(levels = NULL),
                                           CLASS = col_factor(levels = c(1,2,3,4,0)),
                                           CLASS_COARSE = col_factor(levels = c("EPI_FIRST_WAVE","EPI_SECOND_WAVE"))))

# Import csv file for figure 4
figure_4_data <- read_csv("Data/figure_4.csv", 
                          na = c("N/A","NA","#N/A"," ",""),
                          col_types = cols(COUNTRYCODE = col_factor(levels = NULL),
                                           COUNTRY = col_factor(levels = NULL),
                                           T0 = col_date(format = "%Y-%m-%d"),
                                           T0_POP = col_date(format = "%Y-%m-%d"),
                                           date = col_date(format = "%Y-%m-%d"),
                                           CLASS = col_factor(levels = c(1,2,3,4,0)),
                                           CLASS_COARSE = col_factor(levels = c("EPI_FIRST_WAVE","EPI_SECOND_WAVE","EPI_OTHER")),
                                           GOV_C6_RAISED_DATE = col_date(format = "%Y-%m-%d"),
                                           GOV_C6_LOWERED_DATE = col_date(format = "%Y-%m-%d"),
                                           GOV_C6_RAISED_AGAIN_DATE = col_date(format = "%Y-%m-%d")))

# Import csv file for figure 5
figure_5_data <- read_csv("Data/figure_5.csv", 
                          na = c("N/A","NA","#N/A"," ",""),
                          col_types = cols(country = col_factor(levels = NULL),
                                           countrycode = col_factor(levels = NULL),
                                           date = col_date(format = "%Y-%m-%d"),
                                           new_tests = col_double(),
                                           new_tests_smoothed = col_double(),
                                           positive_rate = col_double()))


# Process Data for Figure 2 ----------------------------------------------

# Remove Others class from data
figure_2_data <- subset(figure_2_data,CLASS!=0)

# Aggregate data by class and t_pop
figure_2_agg <- aggregate(figure_2_data[c("stringency_index")],
                          by = list(figure_2_data$CLASS, figure_2_data$t_pop),
                          FUN = mean)
figure_2_agg <- rename(figure_2_agg, c("Group.1"="CLASS", "Group.2"="t_pop"))

# Get the number of elements in each class to work out the t_pop xlim values
figure_2_count <- aggregate(figure_2_data[c("stringency_index")],
                          by = list(figure_2_data$CLASS, figure_2_data$t_pop),
                          FUN = length)
figure_2_count <- rename(figure_2_count, c("Group.1"="CLASS", "Group.2"="t_pop","stringency_index"="n_present"))
figure_2_count_max <- aggregate(figure_2_count[c("n_present")],
                            by = list(figure_2_count$CLASS),
                            FUN = max)
figure_2_count_max <- rename(figure_2_count_max, c("Group.1"="CLASS","n_present"="n_total"))
figure_2_count <- merge(figure_2_count,figure_2_count_max, by="CLASS")
# n_threshold determines where to cut off t_pop xlim values. Only takes t_pop values for which there are >= n_threshold % of the total present for each class
n_threshold = 0.8
figure_2_count <- subset(figure_2_count, n_present>=n_threshold*n_total)
t_min = min(figure_2_count$t_pop)
t_max = max(figure_2_count$t_pop)


# Plot Figure 2 ------------------------------------------------------------

# Set up colour palette
my_palette_1 <- brewer.pal(name="Blues",n=8)[4:8]
my_palette_2 <- brewer.pal(name="Oranges",n=4)[4]

# Figure 2: Line plot of stringency index over time for each country class
figure_2_plot <- (ggplot(figure_2_agg, aes(x = t_pop, y = stringency_index, colour = CLASS)) 
                  + geom_line(size=1,show.legend = FALSE,na.rm=TRUE)
                  + geom_vline(xintercept=0,linetype="dashed", color=my_palette_2, size=1)
                  + annotate("text",x=2,y=97,hjust=0,label="T0 (First Day Surpassing Cumulative 5 Cases per Million)",color=my_palette_2)
                  + theme_light()
                  + coord_cartesian(xlim=c(t_min, t_max))
                  + scale_color_manual(values = my_palette_1, name = "Epidemic Wave State", labels = c("Entering First Wave", "Past First Wave", "Entering Second Wave","Past Second Wave"))
                  + theme(plot.title=element_text(hjust = 0.5), axis.line=element_line(color="black",size=0.7),axis.ticks=element_line(color="black",size=0.7))
                  + scale_x_continuous(breaks=seq(floor(t_min/10)*10,ceiling(t_max/10)*10,10),expand=c(0,0),limits=c(t_min,t_max))
                  + scale_y_continuous(breaks=seq(0,100,10),expand = c(0,0),limits = c(0, 100))
                  + labs(title = "Average Stringency Index Over Time", x = "Days Since T0", y = "Stringency Index"))
figure_2_plot
ggsave("./Plots/figure_2.png", plot = figure_2_plot, width = 12,  height = 7)


# Process Data for Figure 3 ------------------------------------------------

# Remove Others class from data
figure_3_data <- subset(figure_3_data,CLASS!=0)


# Plot Figure 3 ------------------------------------------------------------

# Set up colour palette
my_palette_1 <- brewer.pal(name="Blues",n=8)[4:8]
my_palette_2 <- brewer.pal(name="Oranges",n=4)[4]

# Figure 3: Scatter plot of government response time against number of cases for each country
figure_3_plot <- (ggplot(figure_3_data, aes(x = GOV_MAX_SI_DAYS_FROM_T0_POP, y = EPI_CONFIRMED_PER_10K, colour = CLASS)) 
                  + geom_point(size=1.5,shape=1,alpha=0.9,stroke=1.5)
                  + geom_vline(xintercept=0,linetype="dashed", color=my_palette_2, size=1)
                  + annotate("text",x=2,y=490,hjust=0,label="T0 (First Day Surpassing Cumulative 5 Cases per Million)",color=my_palette_2)
                  # Label countries that have high number of cases, or early/late government response times
                  + geom_text(data=subset(figure_3_data, 
                                          (EPI_CONFIRMED_PER_10K >= quantile(figure_3_data$EPI_CONFIRMED_PER_10K, 0.95)) |
                                          (GOV_MAX_SI_DAYS_FROM_T0_POP >= quantile(figure_3_data$GOV_MAX_SI_DAYS_FROM_T0_POP, 0.95)) |
                                          (GOV_MAX_SI_DAYS_FROM_T0_POP <= quantile(figure_3_data$GOV_MAX_SI_DAYS_FROM_T0_POP, 0.02))),
                              aes(label=COUNTRY),
                              hjust=-0.1, vjust=-0.1,
                              show.legend = FALSE)
                  + theme_light()
                  + theme(plot.title=element_text(hjust = 0.5), axis.line=element_line(color="black",size=0.7),axis.ticks=element_line(color="black",size=0.7))
                  + scale_color_manual(values = my_palette_1, name = "Epidemic Wave State", labels = c("Entering First Wave", "Past First Wave", "Entering Second Wave","Past Second Wave"))
                  + scale_x_continuous(breaks=seq(-125,200,25),expand = c(0,0),limits = c(-125, 200))
                  + scale_y_continuous(breaks=seq(0,500,50),expand = c(0,0),limits = c(0, 500))
                  + labs(title = "Total Confirmed Cases Against Government Response Time", x = "Government Response Time (Days from T0 to Peak of Stringency)", y = "Total Confirmed Cases per 10,000 Population"))
figure_3_plot
ggsave("./Plots/figure_3.png", plot = figure_3_plot, width = 12,  height = 7)


figure_2_and_3 <- grid.arrange(figure_2_plot,figure_3_plot,
                               ncol=2)
ggsave("./Plots/figure_2_and_3.png", plot = figure_2_and_3, width = 20,  height = 7)


# Process Data for Figure 4 ------------------------------------------------


# Remove Others class from data
figure_4_data <- subset(figure_4_data,CLASS%in%c(3,4))

# Aggregate data by t_pop
figure_4_agg <- aggregate(figure_4_data[c("stringency_index","residential_smooth","new_per_day_smooth_per10k")],
                          by = list(figure_4_data$t_pop),
                          FUN = mean,
                          na.rm=TRUE)
figure_4_agg <- rename(figure_4_agg, c("Group.1"="t_pop"))

# Compute average dates of C6 flag
c6_raised = mean(figure_4_data$GOV_C6_RAISED_DATE-figure_4_data$T0_POP,na.rm=TRUE)
c6_lowered = mean(figure_4_data$GOV_C6_LOWERED_DATE-figure_4_data$T0_POP,na.rm=TRUE)
c6_raised_again = mean(figure_4_data$GOV_C6_RAISED_AGAIN_DATE-figure_4_data$T0_POP,na.rm=TRUE)
# Count the number of countries that raised/lowered c6 flag
n_c6_raised = nrow(unique(na.omit(figure_4_data[c("GOV_C6_RAISED_DATE","COUNTRYCODE","T0_POP")])))
n_c6_lowered = nrow(unique(na.omit(figure_4_data[c("GOV_C6_LOWERED_DATE","COUNTRYCODE","T0_POP")])))
n_c6_raised_again = nrow(unique(na.omit(figure_4_data[c("GOV_C6_RAISED_AGAIN_DATE","COUNTRYCODE","T0_POP")])))


# Get the number of elements in each class to work out the t_pop xlim values
figure_4_count <- aggregate(figure_4_data[c("stringency_index","residential_smooth","new_per_day_smooth_per10k")],
                            by = list(figure_4_data$t_pop),
                            FUN = length)
figure_4_count <- rename(figure_4_count, c("Group.1"="t_pop"))
# n_threshold determines where to cut off t_pop xlim values. Only takes t_pop values for which there are >= n_threshold % of the total present for each class
n_threshold = 0.8
figure_4_count <- subset(figure_4_count, 
                         (stringency_index>=n_threshold*max(figure_4_count$stringency_index,na.rm=TRUE))&
                         (residential_smooth>=n_threshold*max(figure_4_count$residential_smooth,na.rm=TRUE))&
                         (new_per_day_smooth_per10k>=n_threshold*max(figure_4_count$new_per_day_smooth_per10k,na.rm=TRUE)))
t_min = min(figure_4_count$t_pop)
t_max = max(figure_4_count$t_pop)

# Select individual countries to show: take top n countries by total confirmed cases
# Get max value of confirmed cases for each country
figure_4_max_confirmed <- aggregate(figure_4_data[c("confirmed")],
                          by = list(figure_4_data$COUNTRY),
                          FUN = max,
                          na.rm=TRUE)
figure_4_max_confirmed <- rename(figure_4_max_confirmed, c("Group.1"="COUNTRY"))
figure_4_max_confirmed <- figure_4_max_confirmed[order(-figure_4_max_confirmed$confirmed),]
top_n_countries <- head(figure_4_max_confirmed, 5)
selected_countries <- c("France","Spain","United States","Romania","South Korea","Australia")
#figure_4_data_indiv <- subset(figure_4_data,COUNTRY%in%top_n_countries$COUNTRY)
figure_4_data_indiv <- subset(figure_4_data,COUNTRY%in%selected_countries)
  
# Normalize countries new per day to be on same scale as aggregate
figure_4_max_new <- aggregate(figure_4_data[c("new_per_day_smooth_per10k")],
                                    by = list(figure_4_data$COUNTRY),
                                    FUN = max,
                                    na.rm=TRUE)
figure_4_max_new <- rename(figure_4_max_new, c("Group.1"="COUNTRY","new_per_day_smooth_per10k"="max_new"))
figure_4_data_indiv <- merge(figure_4_data_indiv,figure_4_max_new, by="COUNTRY")
agg_max <- max(figure_4_agg[(figure_4_agg$t_pop>=t_min)&(figure_4_agg$t_pop<=t_max),"new_per_day_smooth_per10k"],na.rm=TRUE)
figure_4_data_indiv$new_per_day_smooth_per10k_normalized <- figure_4_data_indiv$new_per_day_smooth_per10k*agg_max/figure_4_data_indiv$max_new

# Plot Figure 4 ------------------------------------------------------------

# Set up colour palette
my_palette_1 <- brewer.pal(name="Blues",n=9)[9]
my_palette_2 <- brewer.pal(name="Oranges",n=4)[4]
my_palette_3 <- brewer.pal(name="Blues",n=8)[c(8,2:7)]

text_1 = paste("Stay at home restrictions implemented","\n","(Mean for ",n_c6_raised," countries)",sep="")
text_2 = paste("Restrictions removed","\n","(Mean for ",n_c6_lowered," countries)",sep="")
text_3 = paste("Restrictions re-implemented","\n","(Mean for ",n_c6_raised_again," countries)",sep="")


# Figure 4_1: New Cases per Day (normalized per 10000 population) over time
figure_4_1 <- (ggplot()
                + geom_line(data=figure_4_data_indiv,aes(x=t_pop,y=new_per_day_smooth_per10k_normalized,color=COUNTRY),alpha=0.7, size=1, na.rm=TRUE)
                + geom_line(data=figure_4_agg, aes(x=t_pop,y=new_per_day_smooth_per10k,color="Aggregate"),size=1.5,na.rm=TRUE)
                #+ geom_vline(xintercept=0,linetype="dashed", color=my_palette_2, size=1)
                #+ annotate("text",x=1,y=0.5,hjust=0,vjust=1,label="T0",color=my_palette_2)
                + geom_vline(xintercept=c6_raised,linetype="dashed", color=my_palette_2, size=1)
                + annotate("text",x=c6_raised+1,y=0.58,hjust=0,vjust=1,label=text_1,color=my_palette_2,size=3)
                + geom_vline(xintercept=c6_lowered,linetype="dashed", color=my_palette_2, size=1)
                + annotate("text",x=c6_lowered+1,y=0.58,hjust=0,vjust=1,label=text_2,color=my_palette_2,size=3)
                + geom_vline(xintercept=c6_raised_again,linetype="dashed", color=my_palette_2, size=1)
                + annotate("text",x=c6_raised_again+1,y=0.58,hjust=0,vjust=1,label=text_3,color=my_palette_2,size=3)
                + theme_light()
                + coord_cartesian(xlim=c(t_min, t_max))
                + theme(plot.title=element_text(hjust = 0.5), axis.line=element_line(color="black",size=0.7),axis.ticks=element_line(color="black",size=0.7))
                + scale_x_continuous(breaks=seq(floor(t_min/10)*10,ceiling(t_max/10)*10,10), expand=c(0,0),limits=c(t_min,t_max))
                + scale_y_continuous(expand = c(0,0),limits = c(0, NA))
                + scale_color_manual(values = my_palette_3, name = "Country")
                + labs(x = element_blank(), y = "New Cases per Day per 10,000\n(Individual Countries Normalized)"))
# Figure 4_2: Stringency Index over time
figure_4_2 <- (ggplot()
               + geom_line(data=figure_4_data_indiv,aes(x=t_pop,y=stringency_index,color=COUNTRY),alpha=0.7, size=1, na.rm=TRUE)
               + geom_line(data=figure_4_agg, aes(x=t_pop,y=stringency_index,color="Aggregate"),size=1.5,na.rm=TRUE)
               #+ geom_vline(xintercept=0,linetype="dashed", color=my_palette_2, size=1)
               + geom_vline(xintercept=c6_raised,linetype="dashed", color=my_palette_2, size=1)
               + geom_vline(xintercept=c6_lowered,linetype="dashed", color=my_palette_2, size=1)
               + geom_vline(xintercept=c6_raised_again,linetype="dashed", color=my_palette_2, size=1)
               + theme_light()
               + coord_cartesian(xlim=c(t_min, t_max))
               + theme(plot.title=element_text(hjust = 0.5), axis.line=element_line(color="black",size=0.7),axis.ticks=element_line(color="black",size=0.7))
               + scale_x_continuous(breaks=seq(floor(t_min/10)*10,ceiling(t_max/10)*10,10), expand=c(0,0),limits=c(t_min,t_max))
               + scale_y_continuous(breaks=seq(0,100,20),expand = c(0,0),limits = c(0, 100))
               + scale_color_manual(values = my_palette_3, name = "Country")
               + labs(x = element_blank(), y = "Stringency Index"))
# Figure 4_3: Residential Mobility over time
figure_4_3 <- (ggplot() 
               + geom_line(data=figure_4_data_indiv,aes(x=t_pop,y=residential_smooth,color=COUNTRY),alpha=0.7, size=1, na.rm=TRUE)
               + geom_line(data=figure_4_agg, aes(x=t_pop,y=residential_smooth,color="Aggregate"),size=1.5,na.rm=TRUE)
               #+ geom_vline(xintercept=0,linetype="dashed", color=my_palette_2, size=1)
               + geom_vline(xintercept=c6_raised,linetype="dashed", color=my_palette_2, size=1)
               + geom_vline(xintercept=c6_lowered,linetype="dashed", color=my_palette_2, size=1)
               + geom_vline(xintercept=c6_raised_again,linetype="dashed", color=my_palette_2, size=1)
               + theme_light()
               + coord_cartesian(xlim=c(t_min, t_max))
               + theme(plot.title=element_text(hjust = 0.5), axis.line=element_line(color="black",size=0.7),axis.ticks=element_line(color="black",size=0.7))
               + scale_x_continuous(breaks=seq(floor(t_min/10)*10,ceiling(t_max/10)*10,10), expand=c(0,0),limits=c(t_min,t_max))
               + scale_y_continuous(breaks=seq(0,100,5),expand = c(0,0),limits = c(0, NA))
               + scale_color_manual(values = my_palette_3, name = "Country")
               + labs(x = "Days Since T0 (First Day Surpassing Cumulative 5 Cases per Million)", y = "Change in Residential Mobility from Baseline"))

figure_4_all <- grid.arrange(figure_4_1,figure_4_2,figure_4_3,
                             ncol=1,
                             top="Average Cases, Stringency and Residential Mobility Over Time for Countries in Second Wave")

ggsave("./Plots/figure_4.png", plot = figure_4_all, width = 10,  height = 12)


# Process Data for Figure 5 ------------------------------------------------

# Define which countries to plot
country_a = "United States"
country_b = "Belgium"
country_c = "Australia"

# Get t0 values
t0_a <- figure_4_data[figure_4_data$COUNTRY==country_a,"T0_POP"][[1,1]]
t0_b <- figure_4_data[figure_4_data$COUNTRY==country_b,"T0_POP"][[1,1]]
t0_c <- figure_4_data[figure_4_data$COUNTRY==country_c,"T0_POP"][[1,1]]

# Plot Figure 5 ------------------------------------------------------------

# Set up colour palette
my_palette_1 <- brewer.pal(name="Blues",n=4)[2]
my_palette_2 <- brewer.pal(name="Blues",n=4)[4]
my_palette_3 <- brewer.pal(name="Oranges",n=4)[4]

# Individual subplots
figure_5_a_1 <- (ggplot(subset(figure_5_data,country==country_a)) 
                 + geom_line(aes(x = date, y = new_per_day),size=1,color=my_palette_1,na.rm=TRUE)
                 + geom_line(aes(x = date, y = new_per_day_smooth),size=1,color=my_palette_2,na.rm=TRUE)
                 + geom_vline(xintercept=t0_a,linetype="dashed", color=my_palette_3, size=1)
                 + annotate("text",x=t0_a+5,y=73000,hjust=0,label="T0",color=my_palette_3)
                 + labs(title=country_a, y="New Cases per Day", x=element_blank())
                 + theme_light()
                 + scale_y_continuous(expand = c(0,0),limits = c(0, NA))
                 + theme(plot.title = element_text(hjust = 0.5), axis.line=element_line(color="black",size=0.7),axis.ticks=element_line(color="black",size=0.7),plot.margin=unit(c(0,0,0,0),"pt")))
figure_5_a_2 <- (ggplot(subset(figure_5_data,country==country_a)) 
                 + geom_line(aes(x = date, y = dead_per_day),size=1,color=my_palette_1,na.rm=TRUE)
                 + geom_line(aes(x = date, y = dead_per_day_smooth),size=1,color=my_palette_2,na.rm=TRUE)
                 + geom_vline(xintercept=t0_a,linetype="dashed", color=my_palette_3, size=1)
                 + labs(y="Deaths per Day",x=element_blank())
                 + scale_y_continuous(expand = c(0,0),limits = c(0, NA))
                 + theme_light()
                 + theme(plot.title = element_text(hjust = 0.5), axis.line=element_line(color="black",size=0.7),axis.ticks=element_line(color="black",size=0.7),plot.margin=unit(c(0,0,0,0),"pt")))
figure_5_a_3 <- (ggplot(subset(figure_5_data,country==country_a)) 
                 + geom_line(aes(x = date, y = new_tests),size=1,color=my_palette_1,na.rm=TRUE)
                 + geom_line(aes(x = date, y = new_tests_smoothed),size=1,color=my_palette_2,na.rm=TRUE)
                 + geom_vline(xintercept=t0_a,linetype="dashed", color=my_palette_3, size=1)
                 + labs(y="Tests per Day", x="Date")
                 + scale_y_continuous(expand = c(0,0),limits = c(0, NA))
                 + theme_light()
                 + theme(plot.title = element_text(hjust = 0.5), axis.line=element_line(color="black",size=0.7),axis.ticks=element_line(color="black",size=0.7),plot.margin=unit(c(0,0,0,0),"pt")))
figure_5_b_1 <- (ggplot(subset(figure_5_data,country==country_b))
                 + geom_line(aes(x = date, y = new_per_day),size=1,color=my_palette_1,na.rm=TRUE)
                 + geom_line(aes(x = date, y = new_per_day_smooth),size=1,color=my_palette_2,na.rm=TRUE)
                 + geom_vline(xintercept=t0_b,linetype="dashed", color=my_palette_3, size=1)
                 + annotate("text",x=t0_b+5,y=2200,hjust=0,label="T0",color=my_palette_3)
                 + labs(title=country_b,x=element_blank(),y=element_blank())
                 + scale_y_continuous(expand = c(0,0),limits = c(0, NA))
                 + theme_light()
                 + theme(plot.title = element_text(hjust = 0.5), axis.line=element_line(color="black",size=0.7),axis.ticks=element_line(color="black",size=0.7),plot.margin=unit(c(0,0,0,0),"pt")))
figure_5_b_2 <- (ggplot(subset(figure_5_data,country==country_b)) 
                 + geom_line(aes(x = date, y = dead_per_day),size=1,color=my_palette_1,na.rm=TRUE)
                 + geom_line(aes(x = date, y = dead_per_day_smooth),size=1,color=my_palette_2,na.rm=TRUE)
                 + geom_vline(xintercept=t0_b,linetype="dashed", color=my_palette_3, size=1)
                 + labs(x=element_blank(),y=element_blank())
                 + scale_y_continuous(expand = c(0,0),limits = c(0, NA))
                 + theme_light()
                 + theme(plot.title = element_text(hjust = 0.5), axis.line=element_line(color="black",size=0.7),axis.ticks=element_line(color="black",size=0.7),plot.margin=unit(c(0,0,0,0),"pt")))
figure_5_b_3 <- (ggplot(subset(figure_5_data,country==country_b)) 
                 + geom_line(aes(x = date, y = new_tests),size=1,color=my_palette_1,na.rm=TRUE)
                 + geom_line(aes(x = date, y = new_tests_smoothed),size=1,color=my_palette_2,na.rm=TRUE)
                 + geom_vline(xintercept=t0_b,linetype="dashed", color=my_palette_3, size=1)
                 + labs(x="Date",y=element_blank())
                 + scale_y_continuous(expand = c(0,0),limits = c(0, NA))
                 + theme_light()
                 + theme(plot.title = element_text(hjust = 0.5), axis.line=element_line(color="black",size=0.7),axis.ticks=element_line(color="black",size=0.7),plot.margin=unit(c(0,0,0,0),"pt")))
figure_5_c_1 <- (ggplot(subset(figure_5_data,country==country_c)) 
                 + geom_line(aes(x = date, y = new_per_day),size=1,color=my_palette_1,na.rm=TRUE)
                 + geom_line(aes(x = date, y = new_per_day_smooth),size=1,color=my_palette_2,na.rm=TRUE)
                 + geom_vline(xintercept=t0_c,linetype="dashed", color=my_palette_3, size=1)
                 + annotate("text",x=t0_c+5,y=680,hjust=0,label="T0",color=my_palette_3)
                 + labs(title=country_c,x=element_blank(),y=element_blank())
                 + scale_y_continuous(expand = c(0,0),limits = c(0, NA))
                 + theme_light()
                 + theme(plot.title = element_text(hjust = 0.5), axis.line=element_line(color="black",size=0.7),axis.ticks=element_line(color="black",size=0.7),plot.margin=unit(c(0,0,0,0),"pt")))
figure_5_c_2 <- (ggplot(subset(figure_5_data,country==country_c))
                 + geom_line(aes(x = date, y = dead_per_day),size=1,color=my_palette_1,na.rm=TRUE)
                 + geom_line(aes(x = date, y = dead_per_day_smooth),size=1,color=my_palette_2,na.rm=TRUE)
                 + geom_vline(xintercept=t0_c,linetype="dashed", color=my_palette_3, size=1)
                 + labs(x=element_blank(),y=element_blank())
                 + scale_y_continuous(expand = c(0,0),limits = c(0, NA))
                 + theme_light()
                 + theme(plot.title = element_text(hjust = 0.5), axis.line=element_line(color="black",size=0.7),axis.ticks=element_line(color="black",size=0.7),plot.margin=unit(c(0,0,0,0),"pt")))
figure_5_c_3 <- (ggplot(subset(figure_5_data,country==country_c)) 
                 + geom_line(aes(x = date, y = new_tests),size=1,color=my_palette_1,na.rm=TRUE)
                 + geom_line(aes(x = date, y = new_tests_smoothed),size=1,color=my_palette_2,na.rm=TRUE)
                 + geom_vline(xintercept=t0_c,linetype="dashed", color=my_palette_3, size=1)
                 + labs(x="Date",y=element_blank())
                 + scale_y_continuous(expand = c(0,0),limits = c(0, NA))
                 + theme_light()
                 + theme(plot.title = element_text(hjust = 0.5), axis.line=element_line(color="black",size=0.7),axis.ticks=element_line(color="black",size=0.7),plot.margin=unit(c(0,0,0,0),"pt")))
# Combining subplots
plots <- list(figure_5_a_1,figure_5_b_1,figure_5_c_1,
              figure_5_a_2,figure_5_b_2,figure_5_c_2,
              figure_5_a_3,figure_5_b_3,figure_5_c_3)
grobs <- list()
widths <- list()
for (i in 1:length(plots)){
  grobs[[i]] <- ggplotGrob(plots[[i]])
  widths[[i]] <- grobs[[i]]$widths[2:5]
}
maxwidth <- do.call(grid::unit.pmax, widths)
for (i in 1:length(grobs)){
  grobs[[i]]$widths[2:5] <- as.list(maxwidth)
}
figure_5_all <- do.call("grid.arrange", c(grobs, ncol = 3,top = "Cases, Deaths and Testing Over Time"))

ggsave("./Plots/figure_5.png", plot = figure_5_all, width = 12,  height = 9)



# Testing ------------------------------------------------------------------
run_testing=FALSE
if (run_testing=TRUE) {
  
# Stringency index over time for single country
country = "Japan"
t0=figure_4_data[figure_4_data$COUNTRY==country,"T0"][[1,1]]
(ggplot(figure_2_data[figure_2_data$COUNTRY==country,], aes(x = t_pop, y = stringency_index)) 
  + geom_line(size=1)
  + theme_light()
  + labs(title = paste("Test: Stringency Index Over time for ",country)))


# Positive rate on secondary axis. Assume not necessary for now
max_tests=max(subset(figure_5_data,country==country_c)$new_tests_smoothed,na.rm=TRUE)
figure_5_c_3 <- (ggplot(subset(figure_5_data,country==country_c), aes(x = date, y = new_tests_smoothed)) 
                 + geom_line(size=1)
                 + theme_light()
                 + scale_y_continuous(name = "Tests per Day (Smoothed)", 
                                      sec.axis = sec_axis(~./max_tests, name = "Positive Rate (Smoothed)")))
figure_5_c_3 +  geom_line(aes(x = date, y = positive_rate*max_tests))

# Box plot distribution of T0 date for each class
test_data <- unique(figure_4_data[figure_4_data$CLASS!=0,c("COUNTRY","CLASS","T0_POP")])
test_data <- test_data[order(test_data$T0_POP),]
earliest_n <- head(test_data,2)
latest_n <- tail(test_data,5)

test_plot <- (ggplot(test_data,aes(y=T0_POP,x=CLASS))
              + geom_boxplot(na.rm=TRUE,fill="#6baed6",color="#6baed6",alpha=0.2)
              + geom_point(na.rm=TRUE,shape=1,color="#2171b5",alpha=0.6,size=1,stroke=1.5)
              + labs(y="Date of T0",x="Epidemic Wave State",title="T0 Distribution for Countries in Each Epidemic Wave State")
              + theme_light()
              + scale_x_discrete(labels=c("Entering First Wave","Past First Wave","Entering Second Wave","Past Second Wave"))
              + expand_limits(y=c(as.Date("2020-02-01"),as.Date("2020-09-01")))
              + coord_flip()
              + geom_text(data=subset(test_data, 
                                      (COUNTRY %in% earliest_n$COUNTRY) |
                                      (COUNTRY %in% latest_n$COUNTRY)),
                          aes(label=COUNTRY),
                          hjust=-0.1, vjust=-0.5,color="#6baed6",size=3,
                          show.legend = FALSE)
              + theme(plot.title = element_text(hjust = 0.5), axis.line=element_line(color="black",size=0.7),axis.ticks=element_line(color="black",size=0.7)))
#ggsave("./Plots/test_plot.png", plot = test_plot, width = 8,  height = 4)

}
       
# Epidemetrics - Generate Plots (Figures 2, 3)

# Load Packages, Clear, Sink -------------------------------------------------------

# load packages
package_list <- c("readr","ggplot2","gridExtra","plyr","dplyr","ggsci","RColorBrewer","viridis","sf")
for (package in package_list){
  if (!package %in% installed.packages()){
    install.packages(package)
  }
}
lapply(package_list, require, character.only = TRUE)

# clear workspace
rm(list=ls())

# Import Data -------------------------------------------------------------

# Import csv file for Figure 2a
figure_2a_data <- read_csv("./data/figure_2a.csv", 
                          na = c("N/A","NA","#N/A"," ",""),
                          col_types = cols(COUNTRYCODE = col_factor(levels = NULL),
                                           COUNTRY = col_factor(levels = NULL),
                                           CLASS = col_factor(levels = c(1,2,3,4,0))))
# Import csv file for Figure 2b
figure_2b_data <- read_csv("./data/figure_2b.csv", 
                           na = c("N/A","NA","#N/A"," ",""),
                           col_types = cols(COUNTRYCODE = col_factor(levels = NULL),
                                            COUNTRY = col_factor(levels = NULL),
                                            CLASS = col_factor(levels = c(1,2,3,4,0))))
# Import csv file for figure 2c
figure_2c_data <- read_csv("./data/figure_2c.csv", 
                          na = c("N/A","NA","#N/A"," ",""),
                          col_types = cols(COUNTRYCODE = col_factor(levels = NULL),
                                           COUNTRY = col_factor(levels = NULL),
                                           CLASS = col_factor(levels = c(1,2,3,4,0)),
                                           CLASS_COARSE = col_factor(levels = c("EPI_FIRST_WAVE","EPI_SECOND_WAVE"))))

# Import csv file for figure 3
figure_3_data <- read_csv("./data/figure_3.csv", 
                          na = c("N/A","NA","#N/A"," ",""),
                          col_types = cols(country = col_factor(levels = NULL),
                                           countrycode = col_factor(levels = NULL),
                                           date = col_date(format = "%Y-%m-%d"),
                                           new_tests = col_double(),
                                           new_tests_smoothed = col_double(),
                                           positive_rate = col_double()))

# Countries to label in scatterplot --------------------------------------
label_countries <- c("USA","GBR","ESP","BRA","JAP","IND","ZAF","BEL","AUS")

# Process Data for Figure 2 ----------------------------------------------

# Remove Others class from data
figure_2a_data <- subset(figure_2a_data,CLASS!=0)
figure_2b_data <- subset(figure_2b_data,CLASS!=0)
figure_2c_data <- subset(figure_2c_data,CLASS!=0)

# Aggregate data by class and t_pop
figure_2a_agg <- aggregate(figure_2a_data[c("stringency_index")],
                          by = list(figure_2a_data$CLASS, figure_2a_data$t_pop),
                          FUN = mean)
figure_2a_agg <- plyr::rename(figure_2a_agg, c("Group.1"="CLASS", "Group.2"="t_pop"))

figure_2b_agg <- aggregate(figure_2b_data[c("residential_smooth")],
                           by = list(figure_2b_data$CLASS, figure_2b_data$t_pop),
                           FUN = mean)
figure_2b_agg <- plyr::rename(figure_2b_agg, c("Group.1"="CLASS", "Group.2"="t_pop"))

# Get the number of elements in each class to work out the t_pop xlim values
figure_2a_count <- aggregate(figure_2a_data[c("stringency_index")],
                          by = list(figure_2a_data$CLASS, figure_2a_data$t_pop),
                          FUN = length)
figure_2a_count <- plyr::rename(figure_2a_count, c("Group.1"="CLASS", "Group.2"="t_pop","stringency_index"="n_present"))
figure_2a_count_max <- aggregate(figure_2a_count[c("n_present")],
                            by = list(figure_2a_count$CLASS),
                            FUN = max)
figure_2a_count_max <- plyr::rename(figure_2a_count_max, c("Group.1"="CLASS","n_present"="n_total"))
figure_2a_count <- merge(figure_2a_count,figure_2a_count_max, by="CLASS")

figure_2b_count <- aggregate(figure_2b_data[c("residential_smooth")],
                             by = list(figure_2b_data$CLASS, figure_2b_data$t_pop),
                             FUN = length)
figure_2b_count <- plyr::rename(figure_2b_count, c("Group.1"="CLASS", "Group.2"="t_pop","residential_smooth"="n_present"))
figure_2b_count_max <- aggregate(figure_2b_count[c("n_present")],
                                 by = list(figure_2b_count$CLASS),
                                 FUN = max)
figure_2b_count_max <- plyr::rename(figure_2b_count_max, c("Group.1"="CLASS","n_present"="n_total"))
figure_2b_count <- merge(figure_2b_count,figure_2b_count_max, by="CLASS")

# n_threshold determines where to cut off t_pop xlim values. Only takes t_pop values for which there are >= n_threshold % of the total present for each class
n_threshold = 0.8
figure_2a_count <- subset(figure_2a_count, n_present>=n_threshold*n_total)
figure_2b_count <- subset(figure_2b_count, n_present>=n_threshold*n_total)

t_min = min(figure_2a_count$t_pop, figure_2b_count$t_pop)
t_max = max(figure_2a_count$t_pop, figure_2b_count$t_pop)


# Plot Figure 2 ------------------------------------------------------------
# Set up colour palette
my_palette_1 <- brewer.pal(name="YlGnBu",n=8)[4:8]
my_palette_2 <- brewer.pal(name="Oranges",n=4)[4]

# Figure 2a: Line plot of stringency index over time for each country class
figure_2a <- (ggplot(figure_2a_agg, aes(x = t_pop, y = stringency_index, colour = CLASS)) 
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

# Figure 2b: Line plot of residential mobility over time for each country class
figure_2b <- (ggplot(figure_2b_agg, aes(x = t_pop, y = residential_smooth, colour = CLASS)) 
                  + geom_line(size=1,show.legend = FALSE,na.rm=TRUE)
                  + geom_vline(xintercept=0,linetype="dashed", color=my_palette_2, size=1)
                  + annotate("text",x=2,y=27,hjust=0,label="T0 (First Day Surpassing Cumulative 5 Cases per Million)",color=my_palette_2)
                  + theme_light()
                  + coord_cartesian(xlim=c(t_min, t_max))
                  + scale_color_manual(values = my_palette_1, name = "Epidemic Wave State", labels = c("Entering First Wave", "Past First Wave", "Entering Second Wave","Past Second Wave"))
                  + theme(plot.title=element_text(hjust = 0.5), axis.line=element_line(color="black",size=0.7),axis.ticks=element_line(color="black",size=0.7))
                  + scale_x_continuous(breaks=seq(floor(t_min/10)*10,ceiling(t_max/10)*10,10),expand=c(0,0),limits=c(t_min,t_max))
                  + scale_y_continuous(breaks=seq(0,100,5),expand = c(0,0),limits = c(0, 30))
                  + labs(title = "Average Residential Mobility Over Time", x = "Days Since T0", y = "Residential Mobility (Change from Baseline, Smoothed)"))

# Figure 2c: Scatter plot of government response time against number of cases for each country
figure_2c <- (ggplot(figure_2c_data, aes(x = GOV_MAX_SI_DAYS_FROM_T0_POP, y = EPI_CONFIRMED_PER_10K, colour = CLASS)) 
                  + geom_point(size=1.5,shape=1,alpha=0.9,stroke=1.5)
                  + geom_vline(xintercept=0,linetype="dashed", color=my_palette_2, size=1)
                  + annotate("text",x=2,y=490,hjust=0,label="T0",color=my_palette_2)
                  # Label countries that have high number of cases, or early/late government response times
                  + geom_text(data=subset(figure_2c_data,
                                          (COUNTRYCODE %in% label_countries) |
                                          (EPI_CONFIRMED_PER_10K >= quantile(figure_2c_data$EPI_CONFIRMED_PER_10K, 0.95)) |
                                          (GOV_MAX_SI_DAYS_FROM_T0_POP >= quantile(figure_2c_data$GOV_MAX_SI_DAYS_FROM_T0_POP, 0.95)) |
                                          (GOV_MAX_SI_DAYS_FROM_T0_POP <= quantile(figure_2c_data$GOV_MAX_SI_DAYS_FROM_T0_POP, 0.02))),
                              aes(label=COUNTRY),
                              hjust=-0.1, vjust=-0.1,
                              show.legend = FALSE)
                  + theme_light()
                  + theme(plot.title=element_text(hjust = 0.5), axis.line=element_line(color="black",size=0.7),axis.ticks=element_line(color="black",size=0.7))
                  + scale_color_manual(values = my_palette_1, name = "Epidemic Wave State", labels = c("Entering First Wave", "Past First Wave", "Entering Second Wave","Past Second Wave"))
                  + scale_x_continuous(breaks=seq(-125,200,25),expand = c(0,0),limits = c(-125, 200))
                  + scale_y_continuous(breaks=seq(0,500,50),expand = c(0,0),limits = c(0, 500))
                  + labs(title = "Total Confirmed Cases Against Government Response Time", x = "Government Response Time (Days from T0 to Peak of Stringency)", y = "Total Confirmed Cases per 10,000 Population"))

figure_2_all <- grid.arrange(grobs=list(figure_2a,figure_2b,figure_2c),
                             widths = c(1, 1.2),
                             layout_matrix = rbind(c(1, 3),
                                                   c(2,  3)),
                             top = "Figure 2: Government and Public Response")
ggsave("./plots/figure_2a.png", plot = figure_2a, width = 9,  height = 7)
ggsave("./plots/figure_2b.png", plot = figure_2b, width = 9,  height = 7)
ggsave("./plots/figure_2c.png", plot = figure_2c, width = 9,  height = 7)
ggsave("./plots/figure_2.png", plot = figure_2_all, width = 15,  height = 8)


# Process Data for Figure 5 ------------------------------------------------

# Define which countries to plot
country_a = "United States"
country_b = "Belgium"
country_c = "Australia"

# Plot Figure 5 ------------------------------------------------------------

# Set up colour palette
my_palette_1 <- brewer.pal(name="YlGnBu",n=4)[2]
my_palette_2 <- brewer.pal(name="YlGnBu",n=4)[4]
my_palette_3 <- brewer.pal(name="Oranges",n=4)[4]

# Individual subplots
figure_3_a_1 <- (ggplot(subset(figure_3_data,country==country_a)) 
                 + geom_line(aes(x = date, y = new_per_day),size=1,color=my_palette_1,na.rm=TRUE)
                 + geom_line(aes(x = date, y = new_per_day_smooth),size=1,color=my_palette_2,na.rm=TRUE)
                 + labs(title=country_a, y="New Cases per Day", x=element_blank())
                 + theme_light()
                 + scale_y_continuous(expand = c(0,0),limits = c(0, NA))
                 + theme(plot.title = element_text(hjust = 0.5), axis.line=element_line(color="black",size=0.7),axis.ticks=element_line(color="black",size=0.7),plot.margin=unit(c(0,0,0,0),"pt")))
figure_3_a_2 <- (ggplot(subset(figure_3_data,country==country_a)) 
                 + geom_line(aes(x = date, y = dead_per_day),size=1,color=my_palette_1,na.rm=TRUE)
                 + geom_line(aes(x = date, y = dead_per_day_smooth),size=1,color=my_palette_2,na.rm=TRUE)
                 + labs(y="Deaths per Day",x=element_blank())
                 + scale_y_continuous(expand = c(0,0),limits = c(0, NA))
                 + theme_light()
                 + theme(plot.title = element_text(hjust = 0.5), axis.line=element_line(color="black",size=0.7),axis.ticks=element_line(color="black",size=0.7),plot.margin=unit(c(0,0,0,0),"pt")))

max_tests_a=max(subset(figure_3_data,country==country_a)$new_tests,na.rm=TRUE)
max_positive_rate_a=max(subset(figure_3_data,country==country_a)$positive_rate,na.rm=TRUE)
figure_3_a_3 <- (ggplot(subset(figure_3_data,country==country_a)) 
                 + geom_line(aes(x = date, y = new_tests),size=1,color=my_palette_1,na.rm=TRUE)
                 + geom_line(aes(x = date, y = new_tests_smoothed),size=1,color=my_palette_2,na.rm=TRUE)
                 + geom_line(aes(x = date, y = positive_rate*(max_tests_a/max_positive_rate_a)),color=my_palette_3,na.rm=TRUE)
                 + scale_y_continuous(name = "Tests per Day (Smoothed)", 
                                      expand = c(0,0),limits = c(0, NA),
                                      sec.axis = sec_axis(~./(max_tests_a/max_positive_rate_a), name = element_blank()))
                 + labs(x="Date")
                 + theme_light()
                 + theme(plot.title = element_text(hjust = 0.5), axis.line=element_line(color="black",size=0.7),axis.ticks=element_line(color="black",size=0.7),plot.margin=unit(c(0,0,0,0),"pt"),
                         axis.title.y.left = element_text(color=my_palette_2), axis.title.y.right = element_text(color=my_palette_3)))
figure_3_b_1 <- (ggplot(subset(figure_3_data,country==country_b))
                 + geom_line(aes(x = date, y = new_per_day),size=1,color=my_palette_1,na.rm=TRUE)
                 + geom_line(aes(x = date, y = new_per_day_smooth),size=1,color=my_palette_2,na.rm=TRUE)
                 + labs(title=country_b,x=element_blank(),y=element_blank())
                 + scale_y_continuous(expand = c(0,0),limits = c(0, NA))
                 + theme_light()
                 + theme(plot.title = element_text(hjust = 0.5), axis.line=element_line(color="black",size=0.7),axis.ticks=element_line(color="black",size=0.7),plot.margin=unit(c(0,0,0,0),"pt")))
figure_3_b_2 <- (ggplot(subset(figure_3_data,country==country_b)) 
                 + geom_line(aes(x = date, y = dead_per_day),size=1,color=my_palette_1,na.rm=TRUE)
                 + geom_line(aes(x = date, y = dead_per_day_smooth),size=1,color=my_palette_2,na.rm=TRUE)
                 + labs(x=element_blank(),y=element_blank())
                 + scale_y_continuous(expand = c(0,0),limits = c(0, NA))
                 + theme_light()
                 + theme(plot.title = element_text(hjust = 0.5), axis.line=element_line(color="black",size=0.7),axis.ticks=element_line(color="black",size=0.7),plot.margin=unit(c(0,0,0,0),"pt")))
max_tests_b=max(subset(figure_3_data,country==country_b)$new_tests,na.rm=TRUE)
max_positive_rate_b=max(subset(figure_3_data,country==country_b)$positive_rate,na.rm=TRUE)
figure_3_b_3 <- (ggplot(subset(figure_3_data,country==country_b)) 
                 + geom_line(aes(x = date, y = new_tests),size=1,color=my_palette_1,na.rm=TRUE)
                 + geom_line(aes(x = date, y = new_tests_smoothed),size=1,color=my_palette_2,na.rm=TRUE)
                 + geom_line(aes(x = date, y = positive_rate*(max_tests_b/max_positive_rate_b)),color=my_palette_3,na.rm=TRUE)
                 + scale_y_continuous(name = element_blank(),
                                      expand = c(0,0),limits = c(0, NA),
                                      sec.axis = sec_axis(~./(max_tests_b/max_positive_rate_b), name = element_blank()))
                 + labs(x="Date")
                 + theme_light()
                 + theme(plot.title = element_text(hjust = 0.5), axis.line=element_line(color="black",size=0.7),axis.ticks=element_line(color="black",size=0.7),plot.margin=unit(c(0,0,0,0),"pt"),
                         axis.title.y.left = element_text(color=my_palette_2), axis.title.y.right = element_text(color=my_palette_3)))
figure_3_c_1 <- (ggplot(subset(figure_3_data,country==country_c)) 
                 + geom_line(aes(x = date, y = new_per_day),size=1,color=my_palette_1,na.rm=TRUE)
                 + geom_line(aes(x = date, y = new_per_day_smooth),size=1,color=my_palette_2,na.rm=TRUE)
                 + labs(title=country_c,x=element_blank(),y=element_blank())
                 + scale_y_continuous(expand = c(0,0),limits = c(0, NA))
                 + theme_light()
                 + theme(plot.title = element_text(hjust = 0.5), axis.line=element_line(color="black",size=0.7),axis.ticks=element_line(color="black",size=0.7),plot.margin=unit(c(0,0,0,0),"pt")))
figure_3_c_2 <- (ggplot(subset(figure_3_data,country==country_c))
                 + geom_line(aes(x = date, y = dead_per_day),size=1,color=my_palette_1,na.rm=TRUE)
                 + geom_line(aes(x = date, y = dead_per_day_smooth),size=1,color=my_palette_2,na.rm=TRUE)
                 + labs(x=element_blank(),y=element_blank())
                 + scale_y_continuous(expand = c(0,0),limits = c(0, NA))
                 + theme_light()
                 + theme(plot.title = element_text(hjust = 0.5), axis.line=element_line(color="black",size=0.7),axis.ticks=element_line(color="black",size=0.7),plot.margin=unit(c(0,0,0,0),"pt")))
max_tests_c=max(subset(figure_3_data,country==country_c)$new_tests,na.rm=TRUE)
max_positive_rate_c=max(subset(figure_3_data,country==country_c)$positive_rate,na.rm=TRUE)
figure_3_c_3 <- (ggplot(subset(figure_3_data,country==country_c)) 
                 + geom_line(aes(x = date, y = new_tests),size=1,color=my_palette_1,na.rm=TRUE)
                 + geom_line(aes(x = date, y = new_tests_smoothed),size=1,color=my_palette_2,na.rm=TRUE)
                 + geom_line(aes(x = date, y = positive_rate*(max_tests_c/max_positive_rate_c)),color=my_palette_3,na.rm=TRUE)
                 + scale_y_continuous(name = element_blank(), 
                                      expand = c(0,0),limits = c(0, NA),
                                      sec.axis = sec_axis(~./(max_tests_c/max_positive_rate_c), name = "Positive Rate (Smoothed)"))
                 + labs(x="Date")
                 + theme_light()
                 + theme(plot.title = element_text(hjust = 0.5), axis.line=element_line(color="black",size=0.7),axis.ticks=element_line(color="black",size=0.7),plot.margin=unit(c(0,0,0,0),"pt"),
                         axis.title.y.left = element_text(color=my_palette_2), axis.title.y.right = element_text(color=my_palette_3)))

# Combining subplots
plots <- list(figure_3_a_1,figure_3_b_1,figure_3_c_1,
              figure_3_a_2,figure_3_b_2,figure_3_c_2,
              figure_3_a_3,figure_3_b_3,figure_3_c_3)
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
figure_3_all <- do.call("grid.arrange", c(grobs, ncol = 3,top = "Figure 3: Cases, Deaths and Testing Over Time"))

ggsave("./plots/figure_3.png", plot = figure_3_all, width = 12,  height = 9)



# USA Choropleth ---------------------------------------------------------------
# Process Data for USA time series and choropleth ------------------------------
# Import csv file for figure 4: Time series and choropleth for USA
figure_4_data <- read_delim(file="C:/Users/bryan/OneDrive/Desktop/Epidemetrics R Plots/Data/usa_choropleth.csv",
                            delim=";",
                            na = c("N/A","NA","#N/A"," ",""),
                            col_types = cols(gid = col_factor(levels = NULL),
                                             date = col_date(format = "%Y-%m-%d"),
                                             fips = col_factor(levels = NULL)))
# Remove rows with NA in geometry. Required to convert column to shape object
figure_4_data <- subset(figure_4_data,!is.na(geometry))
# Convert "geometry" column to a sfc shape column 
figure_4_data$geometry <- st_as_sfc(figure_4_data$geometry)
# Convert dataframe to a sf shape object with "geometry" containing the shape information
figure_4_data <- st_sf(figure_4_data)

# Sort by GID and date
figure_4_data <- figure_4_data[order(figure_4_data$gid, figure_4_data$date),]
# Compute new cases per day as difference between daily case total
figure_4_data[-1,"new_cases"] <- diff(figure_4_data$cases)
# Remove first day of each GID as it does not have a value for new cases
figure_4_data <- figure_4_data[duplicated(figure_4_data$gid),]
# Set any negative values for new cases to 0
figure_4_data[figure_4_data$new_cases<0,"new_cases"] <- 0
# Compute new cases per 10000 popuation
figure_4_data$new_cases_per_10k <- 10000*figure_4_data$new_cases/figure_4_data$Population

# Select individual GIDs to show: take top 10 by total confirmed cases
# Get max value of confirmed cases for each country
figure_4_max_confirmed <- aggregate(figure_4_data[c("cases")],
                                    by = list(figure_4_data$gid),
                                    FUN = max,
                                    na.rm=TRUE)
figure_4_max_confirmed <- plyr::rename(figure_4_max_confirmed, c("Group.1"="gid"))
figure_4_max_confirmed <- figure_4_max_confirmed[order(-figure_4_max_confirmed$cases),]
top_n <- head(figure_4_max_confirmed, 5)
figure_4a_data <- subset(figure_4_data,gid%in%top_n$gid)

# Aggregate mean for each day
figure_4a_data_agg <- aggregate(data.frame(figure_4_data[c("new_cases_per_10k")]),
                                by = list(figure_4_data$date),
                                FUN = mean,
                                na.rm=TRUE)
figure_4a_data_agg <- plyr::rename(figure_4a_data_agg, c("Group.1"="date"))

# Define which dates to plot in choropleth
date_1 <- as.Date("2020-04-15")
date_2 <- as.Date("2020-07-15")

figure_4b1_data <- subset(figure_4_data,date==date_1)
figure_4b2_data <- subset(figure_4_data,date==date_2)

color_max <- max(figure_4b1_data$new_cases_per_10k,figure_4b2_data$new_cases_per_10k)

# Plot USA time series and choropleth ----------------------------------------------
# Figure 4a: Time series of cases over time
# Set up colour palette
my_palette_1 <- brewer.pal(name="YlGnBu",n=4)[2]
my_palette_2 <- brewer.pal(name="YlGnBu",n=4)[4]

figure_4a <- (ggplot()
               + geom_line(data = figure_4_data, aes(x=date, y=new_cases_per_10k), color=my_palette_1,size=1,na.rm=TRUE)
               + geom_line(data = figure_4a_data_agg, aes(x=date, y=new_cases_per_10k), color=my_palette_2, size=2, na.rm=TRUE)
               + labs(title="New Cases Over Time for US Counties", y="New Cases per Day per 10,000 Population", x="Date")
               + scale_y_continuous(expand=c(0,0), limits=c(0, 3)) 
               + theme_light()
               + theme(plot.title = element_text(hjust = 0.5), axis.line=element_line(color="black",size=0.7),axis.ticks=element_line(color="black",size=0.7),plot.margin=unit(c(0,0,0,0),"pt")))

# Figure 4b: Choropleth of US counties at particular timestamps

figure_4b1 <- (ggplot(data = figure_4b1_data) 
                + geom_sf(aes(fill=new_cases_per_10k), lwd=0, color=NA)
                + labs(title=paste("New Cases per Day per United States County at",date_1), fill="New Cases per Day\nper 10,000")
                + scale_fill_distiller(palette="GnBu", trans="reverse", limits=c(color_max,0))
                + scale_x_continuous(expand=c(0,0), limits=c(-125, -65)) # coordinates are cropped to exclude Alaska
                + scale_y_continuous(expand=c(0,0), limits=c(24, 50))
                + theme_void()
                + theme(plot.title = element_text(hjust = 0.5), panel.grid.major=element_line(colour = "transparent")))

figure_4b2 <- (ggplot(data = figure_4b2_data) 
               + geom_sf(aes(fill=new_cases_per_10k), lwd=0, color=NA)
               + labs(title=paste("New Cases per Day per United States County at",date_2), fill="New Cases per Day\nper 10,000")
               + scale_fill_distiller(palette="GnBu", trans="reverse", limits=c(color_max,0))
               + scale_x_continuous(expand=c(0,0), limits=c(-125, -65)) # coordinates are cropped to exclude Alaska
               + scale_y_continuous(expand=c(0,0), limits=c(24, 50))
               + theme_void()
               + theme(plot.title = element_text(hjust = 0.5), panel.grid.major=element_line(colour = "transparent")))

figure_4_all <- grid.arrange(grobs=list(figure_4a,figure_4b1,figure_4b2),
                             widths = c(1, 1),
                             layout_matrix = rbind(c(1, 1),
                                                   c(2, 3)),
                             top = "Figure 4: USA Epidemiology Across Geographies and Time")

ggsave("./plots/figure_4a.png", plot = figure_4a, width = 9,  height = 7)
ggsave("./plots/figure_4b1.png", plot = figure_4b1, width = 9,  height = 7)
ggsave("./plots/figure_4b2.png", plot = figure_4b2, width = 9,  height = 7)
ggsave("./plots/figure_4.png", plot = figure_4_all, width = 15,  height = 12)




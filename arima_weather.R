# Directories
setwd("directory")
mydata<- read.csv("mly532.csv")
attach(mydata)
weatherarima <- ts(mydata$maxtp, start = c(1941,11), frequency = 12)
plot(weatherarima,type="l",ylab="Temperature in Celsius")
title("Maximum Air Temperature - Dublin")

# Load libraries
library(MASS)
library(tseries)
library(forecast)

# Plot and convert to ln format
lnweather=log(mydata$maxtp[1:741])
lnweather

# ACF, PACF and Dickey-Fuller Test
acf(lnweather, lag.max=20)
pacf(lnweather, lag.max=20)
adf.test(lnweather)

# Time series and seasonality
weatherarima <- ts(lnweather, start = c(1941,11), frequency = 12)
plot(weatherarima,type="l")
title("Maximum Air Temperature - Dublin")

components <- decompose(weatherarima)
components
plot(components)

# ARIMA
fitlnweather<-auto.arima(weatherarima, trace=TRUE, test="kpss", ic="bic")
fitlnweather
confint(fitlnweather)
plot(weatherarima,type='l')
title('Maximum Air Temperature - Dublin')
exp(lnweather)

# Forecasted Values From ARIMA
forecastedvalues_ln=forecast(fitlnweather,h=186)
forecastedvalues_ln
plot(forecastedvalues_ln)

forecastedvaluesextracted=as.numeric(forecastedvalues_ln$mean)
finalforecastvalues=exp(forecastedvaluesextracted)
finalforecastvalues

# Percentage Error
df<-data.frame(mydata$maxtp[742:927],finalforecastvalues)
col_headings<-c("Actual Weather","Forecasted Weather")
names(df)<-col_headings
attach(df)
percentage_error=((df$`Actual Weather`-df$`Forecasted Weather`)/(df$`Actual Weather`))
percentage_error
mean(percentage_error)
percentage_error=data.frame(abs(percentage_error))
accuracy=data.frame(percentage_error[percentage_error$abs.percentage_error. < 0.1,])
frequency=as.data.frame(table(accuracy))
sum(frequency$Freq)/186
hist(percentage_error$abs.percentage_error.,main="Histogram")

# Ljung-Box
Box.test(fitlnweather$resid, lag=5, type="Ljung-Box")
Box.test(fitlnweather$resid, lag=10, type="Ljung-Box")
Box.test(fitlnweather$resid, lag=15, type="Ljung-Box")

# Simple exponential smoothing with additive errors
fit1 <- ets(weatherarima)
fit1

forecastedvalues_ets=forecast(fit1,h=186)
forecastedvalues_ets
plot(forecastedvalues_ets)
forecastedvaluesextractedets=as.numeric(forecastedvalues_ets$mean)
finalforecastvaluesets=exp(forecastedvaluesextractedets)
finalforecastvaluesets

# Percentage Error
df<-data.frame(mydata$maxtp[742:927],finalforecastvaluesets)
col_headings<-c("Actual Weather","Forecasted Weather")
names(df)<-col_headings
attach(df)
percentage_error=((df$`Actual Weather`-df$`Forecasted Weather`)/(df$`Actual Weather`))
percentage_error
mean(percentage_error)
percentage_error=data.frame(abs(percentage_error))
accuracy=data.frame(percentage_error[percentage_error$abs.percentage_error. < 0.1,])
frequency=as.data.frame(table(accuracy))
sum(frequency$Freq)/186

# SARIMA
fit.1<-Arima(weatherarima, order = c(1,0,0))

# Forecasted Values From ARIMA
forecastedvalues_s=forecast(fit.1,h=186)
forecastedvalues_s
plot(forecastedvalues_s)

forecastedvaluesextracted=as.numeric(forecastedvalues_s$mean)
finalforecastvaluesseason=exp(forecastedvaluesextracted)
finalforecastvaluesseason

# Percentage Error
df<-data.frame(mydata$maxtp[742:927],finalforecastvaluesseason)
col_headings<-c("Actual Weather","Forecasted Weather")
names(df)<-col_headings
attach(df)
percentage_error=((df$`Actual Weather`-df$`Forecasted Weather`)/(df$`Actual Weather`))
percentage_error
mean(percentage_error)
percentage_error=data.frame(abs(percentage_error))
accuracy=data.frame(percentage_error[percentage_error$abs.percentage_error. < 0.1,])
frequency=as.data.frame(table(accuracy))
sum(frequency$Freq)/186

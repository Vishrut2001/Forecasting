#Part 1
library(forecast)
library(imputeTS)

#Data Loading and Cleaning
data <- PCE
View(data)
sum(is.na(data))
imputedts <- na_ma(data$PCE, k = 12, weighting = "exponential")
ets(completets)
PCEdata <- ts(imputedts, start = c(1959,1), end = c(2023,11), frequency = 12)
plot(PCEdata)
seasonplot(PCEdata)
plot(ma(PCEdata,21))
tsdisplay(PCEdata)
end(PCEdata)
#Data split
test_size <- 156
train_data <- subset(PCEdata, end=length(PCEdata)-test_size)
test_data <- subset(PCEdata, start=length(PCEdata)-test_size+1)
length(train_data)
length(test_data)

# holt
holt_model <- holt(train_data, h = 156)
plot(holt_model)
autoplot(holt_model) + autolayer(PCEdata)
holt_oct <- holt(PCEdata, h = 11)
summary(holt_oct)
# seasonal naive
seasonal_model <- snaive(train_data, h = 156, seasonal = 'additive')
plot(seasonal_model)
seasonal_model
autoplot(PCEdata) + autolayer(seasonal_model)
plot(seasonal_model)
plot(holt_model)

# accuracy
accuracy(seasonal_model, PCEdata)
accuracy(holt_model, PCEdata)
#ARIMA
plot(diff(PCEdata))

model_arima <- auto.arima(train_data)
summary(model_arima)
fc_arima <- arima(train_data, order = c(3,2,2))
summary(forecast(fc_arima, h = 156))
accuracy(forecast(fc_arima, h = 156), PCEdata)
plot(forecast(fc_arima, h = 156))
autoplot(forecast(fc_arima, h = 156)) + autolayer(PCEdata)
checkresiduals(fc_arima)

# one step ahead
# Holt 
train_h <- window(PCEdata,end=2010.99)
fit_h <- holt(train)
refit_h <- holt(PCEdata, model=fit_hw)
fc_h <- window(fitted(refit_h), start=2011)
plot(fc_h)
summary(forecast(fc_h, h = 11))
accuracy(forecast(fc_h))
plot(forecast(fc_h , h = 1))
#seasonal naive

train_n <- window(PCEdata,end=2010.99)
fit_n <- snaive(train)
refit_n <- snaive(PCEdata, model=fit_n)
fc_n <- window(fitted(refit_n), start=2011)
plot(fc_n)
summary(forecast(fc_n, h = 11))
accuracy(forecast(fc_n))
#ARIMA

train_a <- window(PCEdata, end = 2010.99)
fit_a <- auto.arima(train)
refit_a <- Arima(PCEdata , model = fit_a)
fc_a <- window(fitted(refit_a), start = 2011)
summary(fc_a)
accuracy(forecast(fc_a))

#Part 2
install.packages("tm")
install.packages("tokenizers")
install.packages("SnowballC")
install.packages("textstem")
install.packages("wordcloud")

library(textcat)
library(tm)
library(tokenizers)
library(textstem)
library(dplyr)
data <- HotelsData
set.seed(067)
sum(is.na(test$Review_score))
data$Text <- data$`Text 1`
data <- data[,-2]
data$language <- lapply(data$Text, textcat)
eng_review <- data[data$language == "english", ]
View(eng_review)
sampled_reviews <- sample_n(eng_review, 2000)


positive_reviews$
  #Splitting by Review
  positive_reviews <- sampled_reviews[sampled_reviews$Review_score >= 4, ]
negative_reviews <- sampled_reviews[sampled_reviews$Review_score <= 2, ]

print(positive_reviews$`Text`[2]) 
#tokenization
words_token_positive <- tokenize_words(positive_reviews$`Text`)
words_token_negative <- tokenize_words(negative_reviews$`Text`)

print(words_token_positive[2])
print(words_token_negative[4])

#corpus formation
corp_positive <- Corpus(VectorSource(positive_reviews$`Text`))
corp_negative <- Corpus(VectorSource(negative_reviews$`Text`))

#tolower
docs_positive <- tm_map(corp_positive, content_transformer(tolower))
print(docs_positive$content[2])
docs_negative<- tm_map(corp_negative, content_transformer(tolower))
print(docs_negative$content[4])

#stopword removal
docs_positive <- tm_map(docs_positive, removeWords, stopwords("english"))
print(docs_positive$content[2])
docs_negative <- tm_map(docs_negative, removeWords, stopwords("english"))
print(docs_negative$content[4])

#Number Removal
docs_positive <- tm_map(docs_positive, content_transformer(removeNumbers))
print(docs_positive$content[2])
docs_negative <- tm_map(docs_negative, content_transformer(removeNumbers))
print(docs_negative$content[4])

# Punctuation removal
docs_positive <- tm_map(docs_positive, content_transformer(removePunctuation))
print(docs_positive$content[2])
docs_negative <- tm_map(docs_negative, content_transformer(removePunctuation))
print(docs_negative$content[4])

#dtm formation positive
dtm_positive <- DocumentTermMatrix(docs_positive)
dtm_positive
findFreqTerms(dtm_positive,180)
dtms_positive <- removeSparseTerms(dtm_positive, 0.99)
dtms_positive
findFreqTerms(dtms_positive, 350)
#word cloud positive
library(wordcloud)
word_freq <- colSums(as.matrix(dtms_positive))
word_freq_df <- data.frame(word = names(word_freq), freq = word_freq)
word_freq_filtered <- word_freq_df[word_freq_df$freq >= 350, ]
wordcloud(words = word_freq_filtered$word, freq = word_freq_filtered$freq, max.words = 100, random.order = FALSE, colors=brewer.pal(1, "Dark2"))


#dtm Negative
dtm_negative <- DocumentTermMatrix(docs_negative)
dtm_negative
findFreqTerms(dtm_negative,80)
dtms_negative <- removeSparseTerms(dtm_negative, 0.99)
dtms_negative
findFreqTerms(dtms_negative, 60)

#wordcloud negative
word_freq_negative <- colSums(as.matrix(dtms_negative))
word_freq_dfn <- data.frame(word = names(word_freq_negative), freq = word_freq_negative)
word_freq_filteredn <- word_freq_dfn[word_freq_dfn$freq >= 60, ]
wordcloud(words = word_freq_filteredn$word, freq = word_freq_filteredn$freq, max.words = 100, random.order = FALSE, colors=brewer.pal(1, "Dark2"))


#Topic Modelling positive
install.packages('ldatuning')
library(ldatuning)
result_positive <- FindTopicsNumber(
  dtm_positive,
  topics = seq(from = 5, to = 20, by = 1),
  metrics = c("Griffiths2004", "CaoJuan2009", "Arun2010"),
  method = "Gibbs",
  control = list(seed = 77),
  mc.cores = 2L,
  verbose = TRUE
)

FindTopicsNumber_plot(result_positive)
# 19 topics for positive

#topic modeling negative

result_negative <- FindTopicsNumber(
  dtm_negative,
  topics = seq(from = 5, to = 20, by = 1),
  metrics = c("Griffiths2004", "CaoJuan2009", "Arun2010"),
  method = "Gibbs",
  control = list(seed = 77),
  mc.cores = 2L,
  verbose = TRUE
)

FindTopicsNumber_plot(result_negative)


# topics for negative 16


#LDA positive
ldaOut_positive <-LDA(dtm_positive,20, method="Gibbs", 
                      control=list(iter=1000,seed=1000))
phi_positive <- posterior(ldaOut_positive)$terms %>% as.matrix

theta_positive <- posterior(ldaOut_positive)$topics %>% as.matrix 

ldaOut.terms_positive <- as.matrix(terms(ldaOut_positive, 20))
lda_terms_positive <- terms(ldaOut_positive, 20)
print(summary(ldaOut_positive))
print(lda_terms_positive)


#LDA Negative
ldaOut_negative <-LDA(dtm_negative,14, method="Gibbs", 
                      control=list(iter=1000,seed=1000))
phi_negative <- posterior(ldaOut_negative)$terms %>% as.matrix

theta_negative <- posterior(ldaOut_negative)$topics %>% as.matrix 

ldaOut.terms_negative <- as.matrix(terms(ldaOut_negative, 20))
lda_terms_negative <- terms(ldaOut_negative, 20)
print(summary(ldaOut_positive))
print(lda_terms_negative)

#Topic probability Positive
ldaOut_positive.topics <- data.frame(topics(ldaOut_positive))
ldaOut_positive.topics$index <- as.numeric(row.names(ldaOut_positive.topics))
positive_reviews$index <- as.numeric(row.names(positive_reviews))
datawithtopic_positive <- merge(positive_reviews, ldaOut_positive.topics, by='index',all.x=TRUE)
datawithtopic_positive <- datawithtopic[order(datawithtopic_positive$index), ]
datawithtopic_positive[0:10,]
topicProbabilities_positive <- as.data.frame(ldaOut_positive@gamma)
topicProbabilities_positive[0:10,1:20]



#Topic probability Negative
ldaOut_negative.topics <- data.frame(topics(ldaOut_negative))
ldaOut_negative.topics$index <- as.numeric(row.names(ldaOut_negative.topics))
negative_reviews$index <- as.numeric(row.names(negative_reviews))
datawithtopic_negative <- merge(negative_reviews, ldaOut_negative.topics, by='index',all.x=TRUE)
datawithtopic_negative <- datawithtopic[order(datawithtopic_negative$index), ]
datawithtopic_negative[0:10,]
topicProbabilities_negative <- as.data.frame(ldaOut_negative@gamma)
topicProbabilities_negative[0:10,1:14]




# Script Header ----
# File-Name:      PH125_9_movielens_capstone_script_mdt.R
# Date:           May 9, 2019                                   
# Author:         Mario De Toma <mdt.datascience@gmail.com>
# Purpose:        R script for submission of PH125_9 movielens capstone project for
#                 HarvardX Data Science Professional Certificate
# Data Used:      MovieLens 10M dataset   
# Packages Used:  dplyr, tidyr, softImpute   

# This program is believed to be free of errors, but it comes with no guarantee! 
# The user bears all responsibility for interpreting the results.

# All source code is copyright (c) 2019, under the Simplified BSD License.  
# For more information on FreeBSD see: http://www.opensource.org/licenses/bsd-license.php

# All images and materials produced by this code are licensed under the Creative Commons 
# Attribution-Share Alike 3.0 United States License: http://creativecommons.org/licenses/by-sa/3.0/us/

# All rights reserved.

#############################################################################################


# session init ----
rm(list=ls())
graphics.off()
#setwd("working directory path")

#  load and partition data script provided by HarvardX ----
# Note: this process could take a couple of minutes

if(!require(tidyverse)) {
  install.packages("tidyverse", repos = "http://cran.us.r-project.org")
  library(tidyverse)
}
if(!require(caret)) {
  install.packages("caret", repos = "http://cran.us.r-project.org")
  library(caret)
}

# MovieLens 10M dataset:
# https://grouplens.org/datasets/movielens/10m/
# http://files.grouplens.org/datasets/movielens/ml-10m.zip

dl <- tempfile()
download.file("http://files.grouplens.org/datasets/movielens/ml-10m.zip", dl)

ratings <- read.table(text = gsub("::", "\t", readLines(unzip(dl, "ml-10M100K/ratings.dat"))),
                      col.names = c("userId", "movieId", "rating", "timestamp"))

movies <- str_split_fixed(readLines(unzip(dl, "ml-10M100K/movies.dat")), "\\::", 3)
colnames(movies) <- c("movieId", "title", "genres")
movies <- as.data.frame(movies) %>% mutate(movieId = as.numeric(levels(movieId))[movieId],
                                           title = as.character(title),
                                           genres = as.character(genres))

movielens <- left_join(ratings, movies, by = "movieId")

# Validation set will be 10% of MovieLens data

set.seed(1)
test_index <- createDataPartition(y = movielens$rating, times = 1, p = 0.1, list = FALSE)
edx <- movielens[-test_index,]
temp <- movielens[test_index,]

# Make sure userId and movieId in validation set are also in edx set

validation <- temp %>% 
  semi_join(edx, by = "movieId") %>%
  semi_join(edx, by = "userId")

# Add rows removed from validation set back into edx set

removed <- anti_join(temp, validation)
edx <- rbind(edx, removed)

rm(dl, ratings, movies, test_index, temp, movielens, removed)


# exploratory data analysis ----
# rating overall distribution
edx %>% group_by(rating) %>% summarise(prop = n()/nrow(edx)) %>% 
  ggplot(aes(rating, prop)) + geom_col()

# rating by movie year
edx %>%  
  mutate(movie_year = as.numeric(str_sub(str_extract(title, '[0-9]{4}\\)$'), 1, 4))) %>% 
  group_by(movie_year) %>%
  summarise(mean_rating = mean(rating)) %>% 
  ggplot(mapping = aes(x = movie_year, y = mean_rating)) +
  geom_point() + geom_line() +
  theme(axis.text.x = element_text(angle = 90))

# rating by year_rated
library(lubridate)
edx %>% 
  mutate(year_rated = year(as_datetime(timestamp))) %>% 
  group_by(year_rated) %>% 
  summarise(mean_rating = mean(rating)) %>% 
  ggplot(mapping = aes(x = year_rated, y = mean_rating)) +
  geom_point() + geom_line() +
  theme(axis.text.x = element_text(angle = 90))

# x.5 rating by year rated
edx %>% filter(rating %in% c(0.5, 1.5, 2.5, 3.5, 4.5)) %>% 
  mutate(year_rated = year(as_datetime(timestamp))) %>%
  group_by(year_rated) %>% 
  summarise(half_rated = n()) %>% 
  arrange(half_rated)

# modeling
# main + group level efect model ----

# evaluation of Reccomender through RMSE
RMSE <- function(true_ratings, predicted_ratings){
  sqrt(mean((true_ratings - predicted_ratings)^2))
}

# modeling effects ----
mu <- mean(edx$rating)
movie_effect <- edx %>% 
  group_by(movieId) %>% 
  summarize(b_movie = mean(rating - mu))
user_effect <- edx %>% 
  left_join(movie_effect, by = 'movieId') %>% 
  group_by(userId) %>% 
  summarize(b_user = mean(rating - mu - b_movie))
year_effect <- edx %>% 
  mutate(year_movie = as.numeric(str_sub(str_extract(title, '[0-9]{4}\\)$'), 1, 4))) %>% 
  left_join(movie_effect, by = 'movieId') %>% 
  left_join(user_effect, by = 'userId') %>% 
  group_by(year_movie) %>% 
  summarise(b_year = mean(rating - mu - b_movie - b_user))

# recommender prediction
predicted_real_ratings <- validation %>% 
  mutate(year_movie = as.numeric(str_sub(str_extract(title, '[0-9]{4}\\)$'), 1, 4))) %>%
  left_join(movie_effect, by = 'movieId') %>% 
  left_join(user_effect, by = 'userId') %>% 
  left_join(year_effect, by = 'year_movie') %>% 
  mutate(predicted_real = mu + b_movie + b_user + b_year) %>% 
  .$predicted_real


# evaluating root mean squared error
rmse_0 <- RMSE(true_ratings = validation$rating, predicted_ratings = predicted_real_ratings)
rmse_results <- tibble(method = 'movie + user + movie_year effects', RMSE = rmse_0)
rmse_results %>% knitr::kable()


# regularization of main + group level model ----
gc()
# search for best lambda
lambdas <- seq(0,10, 0.5)
lambda_df <- map_df(lambdas, function(l) {
  mu <- mean(edx$rating)
  movie_effect <- edx %>% 
    group_by(movieId) %>% 
    summarize(b_movie = sum(rating - mu)/(n() + l))
  user_effect <- edx %>% 
    left_join(movie_effect, by = 'movieId') %>% 
    group_by(userId) %>% 
    summarize(b_user = sum(rating - mu - b_movie)/(n() + l))
  year_effect <- edx %>% 
    mutate(year_movie = as.numeric(str_sub(str_extract(title, '[0-9]{4}\\)$'), 1, 4))) %>% 
    left_join(movie_effect, by = 'movieId') %>% 
    left_join(user_effect, by = 'userId') %>% 
    group_by(year_movie) %>% 
    summarise(b_year = sum(rating - mu - b_movie - b_user)/(n() + l))
  
  predicted_real_ratings <- validation %>% 
    mutate(year_movie = as.numeric(str_sub(str_extract(title, '[0-9]{4}\\)$'), 1, 4))) %>%
    left_join(movie_effect, by = 'movieId') %>% 
    left_join(user_effect, by = 'userId') %>% 
    left_join(year_effect, by = 'year_movie') %>% 
    mutate(predicted_real = mu + b_movie + b_user + b_year) %>% 
    .$predicted_real
  
  rmse <- RMSE(true_ratings = validation$rating, predicted_ratings = predicted_real_ratings)
  tibble(lambda =l, rmse = rmse)
})

lambda_best <- lambda_df$lambda[which.min(lambda_df$rmse)]

# best lambda regularized model prediction
l <- lambda_best
mu <- mean(edx$rating)
movie_effect <- edx %>% 
  group_by(movieId) %>% 
  summarize(b_movie = sum(rating - mu)/(n() + l))
user_effect <- edx %>% 
  left_join(movie_effect, by = 'movieId') %>% 
  group_by(userId) %>% 
  summarize(b_user = sum(rating - mu - b_movie)/(n() + l))
year_effect <- edx %>% 
  mutate(year_movie = as.numeric(str_sub(str_extract(title, '[0-9]{4}\\)$'), 1, 4))) %>% 
  left_join(movie_effect, by = 'movieId') %>% 
  left_join(user_effect, by = 'userId') %>% 
  group_by(year_movie) %>% 
  summarise(b_year = sum(rating - mu - b_movie - b_user)/(n() + l))

predicted_real_ratings <- validation %>% 
  mutate(year_movie = as.numeric(str_sub(str_extract(title, '[0-9]{4}\\)$'), 1, 4))) %>%
  left_join(movie_effect, by = 'movieId') %>% 
  left_join(user_effect, by = 'userId') %>% 
  left_join(year_effect, by = 'year_movie') %>% 
  mutate(predicted_real = mu + b_movie + b_user + b_year) %>% 
  .$predicted_real

# evaluating root mean squared rror
rmse_1 <- RMSE(true_ratings = validation$rating, predicted_ratings = predicted_real_ratings)
rmse_results <- bind_rows(rmse_results, 
                          tibble(method = 'movie + user + movie_year effects regularized', RMSE = rmse_1))
rmse_results %>% knitr::kable()

# #uncomment if you can compute this part
# #at least 32 GB RAM needed
#
# # adding latent factor contribution to the model ----
# gc()
# # user movie rating matrix
# user_movie <- edx %>% select(userId, movieId, rating) %>% 
#   spread(key = movieId, value = rating)
# userId_vec <- user_movie$userId
# user_movie <- user_movie %>% select(-userId)
# movieId_vec <- as.numeric(colnames(user_movie))
# um_matrix <- as.matrix(user_movie)
# 
# # als algo for matrix factorization
# if(!require(softImpute)) {
#   install.packages("softImpute", repos = "http://cran.us.r-project.org")
#   library(softImpute)
# }
# um_matrix_sparse <- as(um_matrix, 'Incomplete')
# rm(um_matrix, user_movie); gc()
# um_matrix_sparse_centered <- biScale(um_matrix_sparse, 
#                                      col.scale=FALSE, row.scale=FALSE,
#                                      maxit = 50, thresh = 1e-05, trace=TRUE)
# rating_fits <- softImpute(um_matrix_sparse_centered, type = "als",
#                           rank.max = 51, lambda = 96, 
#                           trace=TRUE)
# rating_fits$d
# 
# idx_user <- numeric(length = nrow(validation))
# idx_movie <- numeric(length = nrow(validation))
# 
# for (it in 1:nrow(validation)) {
#     idx_user[it] <- which(userId_vec == validation$userId[it])
#     idx_movie[it] <- which(movieId_vec == validation$movieId[it])
# }
# 
# latent_factor_effect <- numeric(length = length(idx_user))
# latent_factor_effect <- impute(object = rating_fits, 
#                                  i = idx_user , j = idx_movie, 
#                                  unscale = FALSE)
# 
# # predict real ratings adding latent factor effect
# predicted_real_ratings_mf <- predicted_real_ratings + latent_factor_effect
# 
# 
# # evaluating root mean squared error
# rmse_2 <- RMSE(true_ratings = validation$rating, predicted_ratings = predicted_real_ratings_mf)
# rmse_results <- bind_rows(rmse_results, tibble(method = 'matrix factorization',
#                                                    RMSE = rmse_2))
# 
# rmse_results %>% knitr::kable()


# end of script ########################################################################################
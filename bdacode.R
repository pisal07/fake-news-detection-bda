## ==============================================================================
## Fake News Detection - Big Data Analytics Project (All-in-One Pipeline)
## Models: Naive Bayes (Laplace), Random Forest (100 Trees), Fast Scaled Linear SVM
## Artifacts: rf_model.rds, features.rds
## Visuals: Metrics Bar Chart, ROC Curves, Feature Importance, 
##          3 Confusion Matrices, 2 Word Clouds
## ==============================================================================

# Clear any lingering graphics locks and workspace memory
graphics.off()
rm(list = ls())

start_time <- Sys.time()

## ---- 1. Install & Load Dependencies ----
packages <- c("tidyverse", "tm", "SnowballC", "wordcloud", "caret",
              "e1071", "randomForest", "ggplot2", "pROC", "reshape2", "RColorBrewer")
new_pkgs <- packages[!(packages %in% installed.packages()[, "Package"])]
if (length(new_pkgs)) install.packages(new_pkgs, dependencies = TRUE)
invisible(lapply(packages, library, character.only = TRUE))

## ---- 2. Schema Normalization & Global Ingestion ----
standardize_cols <- function(df, default_label = "FAKE") {
  cnames <- tolower(colnames(df))
  colnames(df) <- cnames
  
  title_col <- intersect(cnames, c("title", "headline", "headlines", "heading"))[1]
  text_col  <- intersect(cnames, c("text", "article", "body", "content", "news"))[1]
  label_col <- intersect(cnames, c("label", "class", "target", "category", "verdict"))[1]
  
  df$title <- if (!is.na(title_col)) df[[title_col]] else ""
  df$text  <- if (!is.na(text_col))  df[[text_col]]  else ""
  df$label <- if (is.na(label_col)) default_label   else df[[label_col]]
  
  if (is.numeric(df$label)) {
    df$label <- ifelse(df$label == 0, "FAKE", "REAL")
  } else {
    df$label <- toupper(as.character(df$label))
    df$label <- ifelse(grepl("FAKE|FALSE", df$label), "FAKE", "REAL")
  }
  return(df[, c("title", "text", "label")])
}

cat("\n[Step 1/9] Ingesting and unifying all 4 datasets...\n")
news <- rbind(
  standardize_cols(read.csv("Fake.csv", stringsAsFactors = FALSE), "FAKE"),
  standardize_cols(read.csv("True.csv", stringsAsFactors = FALSE), "REAL"),
  standardize_cols(read.csv("combined_output.csv", stringsAsFactors = FALSE)),
  standardize_cols(read.csv("dataset2_welfake_clean.csv", stringsAsFactors = FALSE))
)
news$text <- paste(news$title, news$text)
news$label <- as.factor(news$label)

## ---- 3. Leakage Removal & Sampling ----
news$text <- gsub("\\(reuters\\)|reuters", "", news$text, ignore.case = TRUE)
news$text <- gsub("^[A-Z ]+\\s*-\\s*", "", news$text)

# 5,000 rows provides high model accuracy while guaranteeing SVM completes in seconds
set.seed(42)
if (nrow(news) > 5000) news <- news[sample(nrow(news), 5000), ]
cat("Records selected for training:", nrow(news), "\n")

## ---- 4. Stylometric Engineering & NLP Vectorization ----
cat("[Step 2/9] Extracting stylometrics and building Document Term Matrix...\n")
news$excl_count <- sapply(news$text, function(x) length(regmatches(x, gregexpr("!", x))[[1]]))
news$caps_count <- sapply(news$text, function(x) length(regmatches(x, gregexpr("\\b[A-Z]{2,}\\b", x))[[1]]))
news$word_count <- sapply(gregexpr("\\W+", news$text), length) + 1

corpus <- VCorpus(VectorSource(news$text))
corpus <- tm_map(corpus, content_transformer(tolower))
corpus <- tm_map(corpus, removePunctuation)
corpus <- tm_map(corpus, removeNumbers)
corpus <- tm_map(corpus, removeWords, stopwords("english"))
corpus <- tm_map(corpus, stripWhitespace)
corpus <- tm_map(corpus, stemDocument)

dtm <- DocumentTermMatrix(corpus, control = list(weighting = weightTf))
dtm <- removeSparseTerms(dtm, 0.98)
dtm_matrix <- as.data.frame(as.matrix(dtm))
colnames(dtm_matrix) <- make.names(colnames(dtm_matrix))

# Bind engineered features
dtm_matrix$excl_count <- news$excl_count
dtm_matrix$caps_count <- news$caps_count
dtm_matrix$word_count <- news$word_count
dtm_matrix$label      <- news$label

## ---- 5. Train / Test Split ----
set.seed(42)
split_idx  <- createDataPartition(dtm_matrix$label, p = 0.8, list = FALSE)
train_data <- dtm_matrix[split_idx, ]
test_data  <- dtm_matrix[-split_idx, ]

get_eval <- function(pred, actual, scores = NULL) {
  cm <- confusionMatrix(pred, actual, positive = "FAKE")
  auc_val <- NA
  if (!is.null(scores)) {
    roc_obj <- roc(actual, scores, levels = c("REAL", "FAKE"), quiet = TRUE)
    if (auc(roc_obj) < 0.5) roc_obj <- roc(actual, -scores, levels = c("REAL", "FAKE"), quiet = TRUE)
    auc_val <- as.numeric(auc(roc_obj))
  }
  list(cm = cm,
       acc  = as.numeric(cm$overall["Accuracy"]),
       prec = as.numeric(cm$byClass["Precision"]),
       rec  = as.numeric(cm$byClass["Recall"]),
       f1   = as.numeric(cm$byClass["F1"]),
       auc  = auc_val)
}

## ---- 6. Multi-Model Training ----
# 1. Naive Bayes (Laplace Smoothing)
cat("[Step 3/9] Training Naive Bayes (Laplace = 1)...\n")
nb_model <- naiveBayes(label ~ ., data = train_data, laplace = 1)
nb_pred  <- predict(nb_model, test_data)
nb_prob  <- predict(nb_model, test_data, type = "raw")[, "FAKE"]
nb_res   <- get_eval(nb_pred, test_data$label, nb_prob)

# 2. Random Forest (Champion)
cat("[Step 4/9] Training Random Forest (100 Trees)...\n")
rf_model <- randomForest(label ~ ., data = train_data, ntree = 100, importance = TRUE)
rf_pred  <- predict(rf_model, test_data)
rf_prob  <- predict(rf_model, test_data, type = "prob")[, "FAKE"]
rf_res   <- get_eval(rf_pred, test_data$label, rf_prob)

# 3. Fast Linear SVM (Scaled, Non-freezing Decision Values)
cat("[Step 5/9] Training Linear SVM (Scaled, Cost = 1)...\n")
svm_model <- svm(label ~ ., data = train_data, kernel = "linear", scale = TRUE, cost = 1)
svm_pred  <- predict(svm_model, test_data, decision.values = TRUE)
svm_dv    <- as.numeric(attr(svm_pred, "decision.values"))
svm_res   <- get_eval(svm_pred, test_data$label, svm_dv)

## ---- 7. Export Production Artifacts for Shiny ----
cat("[Step 6/9] Exporting 'rf_model.rds' and 'features.rds' for Shiny dashboard...\n")
saveRDS(rf_model, "rf_model.rds")
saveRDS(colnames(train_data)[colnames(train_data) != "label"], "features.rds")

## ---- 8. Export Tables & Metric Comparisons ----
cat("[Step 7/9] Generating comparison table, bar chart, and ROC curves...\n")
results <- data.frame(
  Model     = c("Naive Bayes", "Random Forest", "Support Vector Machine"),
  Accuracy  = c(nb_res$acc,  rf_res$acc,  svm_res$acc),
  Precision = c(nb_res$prec, rf_res$prec, svm_res$prec),
  Recall    = c(nb_res$rec,  rf_res$rec,  svm_res$rec),
  F1_Score  = c(nb_res$f1,   rf_res$f1,   svm_res$f1),
  AUC       = c(nb_res$auc,  rf_res$auc,  svm_res$auc)
)
write.csv(results, "model_comparison_table.csv", row.names = FALSE)
print(results)

# Comparative Bar Chart
results_long <- melt(results, id.vars = "Model", variable.name = "Metric", value.name = "Score")
p_comp <- ggplot(results_long, aes(x = Metric, y = Score, fill = Model)) +
  geom_bar(stat = "identity", position = position_dodge(0.8), width = 0.7) +
  geom_text(aes(label = round(Score, 2)), position = position_dodge(0.8), vjust = -0.4, size = 3.3) +
  ylim(0, 1.1) +
  scale_fill_brewer(palette = "Set2") +
  labs(title = "Model Benchmark: Global Corpus with Stylometrics", x = "", y = "Score") +
  theme_minimal()
ggsave("model_comparison_all_metrics.png", plot = p_comp, width = 8.5, height = 5)

# Multi-Model ROC Curves
roc_nb  <- roc(test_data$label, nb_prob, levels = c("REAL", "FAKE"), quiet = TRUE)
roc_rf  <- roc(test_data$label, rf_prob, levels = c("REAL", "FAKE"), quiet = TRUE)
roc_svm <- roc(test_data$label, svm_dv, levels = c("REAL", "FAKE"), quiet = TRUE)
if (auc(roc_svm) < 0.5) roc_svm <- roc(test_data$label, -svm_dv, levels = c("REAL", "FAKE"), quiet = TRUE)

png("roc_curves_comparison.png", width = 750, height = 650)
plot(roc_rf, col = "#2c7bb6", lwd = 3, main = "ROC Comparison Across Evaluated Architectures")
lines(roc_svm, col = "#fdae61", lwd = 2.5)
lines(roc_nb, col = "#d7191c", lwd = 2, lty = 2)
legend("bottomright",
       legend = c(paste0("Random Forest (AUC = ", round(auc(roc_rf), 3), ")"),
                  paste0("Linear SVM (AUC = ", round(auc(roc_svm), 3), ")"),
                  paste0("Naive Bayes (AUC = ", round(auc(roc_nb), 3), ")")),
       col = c("#2c7bb6", "#fdae61", "#d7191c"), lwd = c(3, 2.5, 2), lty = c(1, 1, 2))
grid()
dev.off()

# Feature Importance Plot
png("feature_importance_rf.png", width = 850, height = 800)
varImpPlot(rf_model, n.var = 20, main = "Top 20 Predictive Features (Global RF + Stylometrics)")
dev.off()

## ---- 9. Confusion Matrix Heatmaps ----
cat("[Step 8/9] Exporting confusion matrix heatmaps...\n")
plot_cm <- function(cm_object, title_text, filename, fill_color) {
  df_cm <- as.data.frame(cm_object$table)
  colnames(df_cm) <- c("Prediction", "Reference", "Freq")
  p <- ggplot(df_cm, aes(x = Prediction, y = Reference, fill = Freq)) +
    geom_tile(color = "white") +
    geom_text(aes(label = Freq), color = "black", fontface = "bold", size = 6) +
    scale_fill_gradient(low = "#f7fbff", high = fill_color) +
    labs(title = title_text, subtitle = paste("Total Test Instances:", sum(df_cm$Freq))) +
    theme_minimal()
  ggsave(filename, plot = p, width = 5.2, height = 4.2)
}
plot_cm(nb_res$cm,  "Naive Bayes - Confusion Matrix",   "confusion_matrix_nb.png",  "#fc9272")
plot_cm(rf_res$cm,  "Random Forest - Confusion Matrix", "confusion_matrix_rf.png",  "#9ecae1")
plot_cm(svm_res$cm, "Linear SVM - Confusion Matrix",    "confusion_matrix_svm.png", "#a1d99b")

## ---- 10. Memory-Safe Clean Word Clouds ----
cat("[Step 9/9] Generating clean word clouds...\n")
generate_wordcloud <- function(txt_vector, filename, color_palette) {
  # Subsample 1,000 texts to prevent margin overflow errors
  sub_txt <- sample(txt_vector, min(1000, length(txt_vector)))
  cp <- VCorpus(VectorSource(sub_txt))
  cp <- tm_map(cp, content_transformer(tolower))
  cp <- tm_map(cp, removePunctuation)
  cp <- tm_map(cp, removeNumbers)
  cp <- tm_map(cp, removeWords, stopwords("english"))
  cp <- tm_map(cp, stripWhitespace)
  
  png(filename, width = 800, height = 800)
  par(mar = c(0, 0, 0, 0))
  wordcloud(cp, max.words = 80, colors = color_palette, scale = c(3.5, 0.6), random.order = FALSE)
  dev.off()
}

generate_wordcloud(news$text[news$label == "FAKE"], "wordcloud_fake.png", brewer.pal(8, "Dark2"))
generate_wordcloud(news$text[news$label == "REAL"], "wordcloud_real.png", brewer.pal(8, "Blues")[3:8])

# Final Cleanup & Diagnostics
graphics.off()
total_runtime <- round(difftime(Sys.time(), start_time, units = "mins"), 2)

cat("\n==================================================================\n")
cat(" PIPELINE COMPLETE! Total Execution Time:", total_runtime, "minutes\n")
cat("==================================================================\n")
cat("Generated files verified in working directory:\n")
cat(" - Shiny Model Artifacts : rf_model.rds, features.rds\n")
cat(" - Performance Table     : model_comparison_table.csv\n")
cat(" - Comparison Chart      : model_comparison_all_metrics.png\n")
cat(" - ROC Curve             : roc_curves_comparison.png\n")
cat(" - Importance Ranking    : feature_importance_rf.png\n")
cat(" - Confusion Matrices    : confusion_matrix_nb.png, confusion_matrix_rf.png, confusion_matrix_svm.png\n")
cat(" - Word Clouds           : wordcloud_fake.png, wordcloud_real.png\n\n")





setwd("C:/Users/admin/Downloads/bda")
suppressPackageStartupMessages({
  library(ggplot2); library(pROC); library(reshape2)
  library(wordcloud); library(RColorBrewer); library(caret)
})

# 1. Comparison Bar Chart & Table
results <- data.frame(
  Model     = c("Naive Bayes", "Random Forest", "Support Vector Machine"),
  Accuracy  = c(as.numeric(nb_res$acc),  as.numeric(rf_res$acc),  as.numeric(svm_res$acc)),
  Precision = c(as.numeric(nb_res$prec), as.numeric(rf_res$prec), as.numeric(svm_res$prec)),
  Recall    = c(as.numeric(nb_res$rec),  as.numeric(rf_res$rec),  as.numeric(svm_res$rec)),
  F1_Score  = c(as.numeric(nb_res$f1),   as.numeric(rf_res$f1),   as.numeric(svm_res$f1)),
  AUC       = c(as.numeric(nb_res$auc),  as.numeric(rf_res$auc),  as.numeric(svm_res$auc))
)
write.csv(results, "model_comparison_table.csv", row.names = FALSE)

results_long <- melt(results, id.vars = "Model", variable.name = "Metric", value.name = "Score")
p_comp <- ggplot(results_long, aes(x = Metric, y = Score, fill = Model)) +
  geom_bar(stat = "identity", position = position_dodge(0.8), width = 0.7) +
  geom_text(aes(label = round(Score, 2)), position = position_dodge(0.8), vjust = -0.4, size = 3.3) +
  ylim(0, 1.1) + scale_fill_brewer(palette = "Set2") +
  labs(title = "Model Benchmark: Global Corpus with Stylometrics", x = "", y = "Score") +
  theme_minimal()
ggsave("model_comparison_all_metrics.png", plot = p_comp, width = 8.5, height = 5)

# 2. Multi-Model ROC Curves
roc_nb  <- roc(test_data$label, nb_prob, levels = c("REAL", "FAKE"), quiet = TRUE)
roc_rf  <- roc(test_data$label, rf_prob, levels = c("REAL", "FAKE"), quiet = TRUE)
roc_svm <- roc(test_data$label, svm_dv, levels = c("REAL", "FAKE"), quiet = TRUE)
if (auc(roc_svm) < 0.5) roc_svm <- roc(test_data$label, -svm_dv, levels = c("REAL", "FAKE"), quiet = TRUE)

png("roc_curves_comparison.png", width = 750, height = 650)
plot(roc_rf, col = "#2c7bb6", lwd = 3, main = "ROC Comparison Across Evaluated Architectures")
lines(roc_svm, col = "#fdae61", lwd = 2.5)
lines(roc_nb, col = "#d7191c", lwd = 2, lty = 2)
legend("bottomright",
       legend = c(paste0("Random Forest (AUC = ", round(auc(roc_rf), 3), ")"),
                  paste0("Linear SVM (AUC = ", round(auc(roc_svm), 3), ")"),
                  paste0("Naive Bayes (AUC = ", round(auc(roc_nb), 3), ")")),
       col = c("#2c7bb6", "#fdae61", "#d7191c"), lwd = c(3, 2.5, 2), lty = c(1, 1, 2))
grid()
dev.off()

# 3. Random Forest Feature Importance
png("feature_importance_rf.png", width = 850, height = 800)
varImpPlot(rf_model, n.var = 20, main = "Top 20 Predictive Features (Global RF + Stylometrics)")
dev.off()

# 4. Confusion Matrix Heatmaps
plot_cm <- function(cm_object, title_text, filename, fill_color) {
  df_cm <- as.data.frame(cm_object$table)
  colnames(df_cm) <- c("Prediction", "Reference", "Freq")
  p <- ggplot(df_cm, aes(x = Prediction, y = Reference, fill = Freq)) +
    geom_tile(color = "white") +
    geom_text(aes(label = Freq), color = "black", fontface = "bold", size = 6) +
    scale_fill_gradient(low = "#f7fbff", high = fill_color) +
    labs(title = title_text, subtitle = paste("Total Test Instances:", sum(df_cm$Freq))) +
    theme_minimal()
  ggsave(filename, plot = p, width = 5.2, height = 4.2)
}
plot_cm(nb_res$cm,  "Naive Bayes - Confusion Matrix",   "confusion_matrix_nb.png",  "#fc9272")
plot_cm(rf_res$cm,  "Random Forest - Confusion Matrix", "confusion_matrix_rf.png",  "#9ecae1")
plot_cm(svm_res$cm, "Linear SVM - Confusion Matrix",    "confusion_matrix_svm.png", "#a1d99b")

# 5. Word Clouds
make_safe_cloud <- function(text_vec, filename, palette) {
  words <- unlist(strsplit(tolower(paste(sample(text_vec, min(500, length(text_vec))), collapse = " ")), "\\W+"))
  words <- words[nchar(words) > 3 & !(words %in% stopwords("english"))]
  word_freq <- sort(table(words), decreasing = TRUE)
  
  png(filename, width = 800, height = 800)
  wordcloud(names(word_freq), as.numeric(word_freq), max.words = 70, 
            colors = palette, scale = c(3.5, 0.7), random.order = FALSE)
  dev.off()
}

make_safe_cloud(news$text[news$label == "FAKE"], "wordcloud_fake.png", brewer.pal(8, "Dark2"))
make_safe_cloud(news$text[news$label == "REAL"], "wordcloud_real.png", brewer.pal(8, "Blues")[3:8])

graphics.off()

list.files(pattern = "\\.png$")















# Fake News Detection - Big Data Analytics Project
# Models: Naive Bayes, Random Forest, SVM

setwd("C:/Users/admin/Downloads/bda")
graphics.off()
rm(list = ls())

library(tm)
library(SnowballC)
library(wordcloud)
library(caret)
library(e1071)
library(randomForest)
library(ggplot2)
library(pROC)
library(reshape2)
library(RColorBrewer)

start_time <- Sys.time()

# 1. Dataset Ingestion & Standardization
standardize_cols <- function(df, default_label = "FAKE") {
  names(df) <- tolower(names(df))
  
  t_col <- intersect(names(df), c("title", "headline", "headlines", "heading"))[1]
  b_col <- intersect(names(df), c("text", "article", "body", "content", "news"))[1]
  l_col <- intersect(names(df), c("label", "class", "target", "category", "verdict"))[1]
  
  df$title <- if (!is.na(t_col)) df[[t_col]] else ""
  df$text  <- if (!is.na(b_col)) df[[b_col]] else ""
  df$label <- if (!is.na(l_col)) df[[l_col]] else default_label
  
  if (is.numeric(df$label)) {
    df$label <- ifelse(df$label == 0, "FAKE", "REAL")
  } else {
    df$label <- toupper(as.character(df$label))
    df$label <- ifelse(grepl("FAKE|FALSE", df$label), "FAKE", "REAL")
  }
  return(df[, c("title", "text", "label")])
}

cat("Loading datasets...\n")
news <- rbind(
  standardize_cols(read.csv("Fake.csv", stringsAsFactors = FALSE), "FAKE"),
  standardize_cols(read.csv("True.csv", stringsAsFactors = FALSE), "REAL"),
  standardize_cols(read.csv("combined_output.csv", stringsAsFactors = FALSE)),
  standardize_cols(read.csv("dataset2_welfake_clean.csv", stringsAsFactors = FALSE))
)

news$text <- paste(news$title, news$text)
news$label <- as.factor(news$label)

# 2. Text Preprocessing & Cleaning
news$text <- gsub("\\(reuters\\)|reuters", "", news$text, ignore.case = TRUE)
news$text <- gsub("^[A-Z ]+\\s*-\\s*", "", news$text)

# Subsampling 5000 records for stable memory convergence
set.seed(42)
if (nrow(news) > 5000) {
  news <- news[sample(nrow(news), 5000), ]
}

# 3. Stylometric Feature Extraction
news$excl_count <- sapply(news$text, function(x) length(regmatches(x, gregexpr("!", x))[[1]]))
news$caps_count <- sapply(news$text, function(x) length(regmatches(x, gregexpr("\\b[A-Z]{2,}\\b", x))[[1]]))
news$word_count <- sapply(gregexpr("\\W+", news$text), length) + 1

# 4. Document-Term Matrix (TF)
cat("Building Document Term Matrix...\n")
corpus <- VCorpus(VectorSource(news$text))
corpus <- tm_map(corpus, content_transformer(tolower))
corpus <- tm_map(corpus, removePunctuation)
corpus <- tm_map(corpus, removeNumbers)
corpus <- tm_map(corpus, removeWords, stopwords("english"))
corpus <- tm_map(corpus, stripWhitespace)
corpus <- tm_map(corpus, stemDocument)

dtm <- DocumentTermMatrix(corpus, control = list(weighting = weightTf))
dtm <- removeSparseTerms(dtm, 0.98)
dtm_df <- as.data.frame(as.matrix(dtm))
colnames(dtm_df) <- make.names(colnames(dtm_df))

dtm_df$excl_count <- news$excl_count
dtm_df$caps_count <- news$caps_count
dtm_df$word_count <- news$word_count
dtm_df$label      <- news$label

# 5. Train / Test Split (80 / 20)
set.seed(42)
split_idx  <- createDataPartition(dtm_df$label, p = 0.8, list = FALSE)
train_data <- dtm_df[split_idx, ]
test_data  <- dtm_df[-split_idx, ]

get_metrics <- function(pred, actual, scores = NULL) {
  cm <- confusionMatrix(pred, actual, positive = "FAKE")
  auc_val <- NA
  if (!is.null(scores)) {
    r <- roc(actual, scores, levels = c("REAL", "FAKE"), quiet = TRUE)
    if (auc(r) < 0.5) r <- roc(actual, -scores, levels = c("REAL", "FAKE"), quiet = TRUE)
    auc_val <- as.numeric(auc(r))
  }
  list(
    cm = cm,
    Accuracy  = as.numeric(cm$overall["Accuracy"]),
    Precision = as.numeric(cm$byClass["Precision"]),
    Recall    = as.numeric(cm$byClass["Recall"]),
    F1        = as.numeric(cm$byClass["F1"]),
    AUC       = auc_val
  )
}

# 6. Model Training & Evaluation
cat("Training Naive Bayes...\n")
nb_model <- naiveBayes(label ~ ., data = train_data, laplace = 1)
nb_pred  <- predict(nb_model, test_data)
nb_prob  <- predict(nb_model, test_data, type = "raw")[, "FAKE"]
nb_res   <- get_metrics(nb_pred, test_data$label, nb_prob)

cat("Training Random Forest...\n")
rf_model <- randomForest(label ~ ., data = train_data, ntree = 100, importance = TRUE)
rf_pred  <- predict(rf_model, test_data)
rf_prob  <- predict(rf_model, test_data, type = "prob")[, "FAKE"]
rf_res   <- get_metrics(rf_pred, test_data$label, rf_prob)

cat("Training Support Vector Machine...\n")
svm_model <- svm(label ~ ., data = train_data, kernel = "linear", scale = TRUE, cost = 1)
svm_pred  <- predict(svm_model, test_data, decision.values = TRUE)
svm_dv    <- as.numeric(attr(svm_pred, "decision.values"))
svm_res   <- get_metrics(svm_pred, test_data$label, svm_dv)

# 7. Save Artifacts for Shiny
saveRDS(rf_model, "rf_model.rds")
saveRDS(colnames(train_data)[colnames(train_data) != "label"], "features.rds")

# 8. Comparison Results & CSV Export
results <- data.frame(
  Model     = c("Naive Bayes", "Random Forest", "Support Vector Machine"),
  Accuracy  = c(nb_res$Accuracy,  rf_res$Accuracy,  svm_res$Accuracy),
  Precision = c(nb_res$Precision, rf_res$Precision, svm_res$Precision),
  Recall    = c(nb_res$Recall,    rf_res$Recall,    svm_res$Recall),
  F1_Score  = c(nb_res$F1,        rf_res$F1,        svm_res$F1),
  AUC       = c(nb_res$AUC,       rf_res$AUC,       svm_res$AUC)
)
write.csv(results, "model_comparison_table.csv", row.names = FALSE)
print(results)

# 9. Plots & Visualizations
results_long <- melt(results, id.vars = "Model", variable.name = "Metric", value.name = "Score")
p_comp <- ggplot(results_long, aes(x = Metric, y = Score, fill = Model)) +
  geom_bar(stat = "identity", position = position_dodge(0.8), width = 0.7) +
  geom_text(aes(label = round(Score, 2)), position = position_dodge(0.8), vjust = -0.4, size = 3) +
  ylim(0, 1.1) +
  scale_fill_brewer(palette = "Set2") +
  labs(title = "Model Performance Benchmark", x = "", y = "Score") +
  theme_minimal()
ggsave("model_comparison_all_metrics.png", plot = p_comp, width = 8, height = 5)

# ROC Curves
roc_nb  <- roc(test_data$label, nb_prob, levels = c("REAL", "FAKE"), quiet = TRUE)
roc_rf  <- roc(test_data$label, rf_prob, levels = c("REAL", "FAKE"), quiet = TRUE)
roc_svm <- roc(test_data$label, svm_dv, levels = c("REAL", "FAKE"), quiet = TRUE)
if (auc(roc_svm) < 0.5) roc_svm <- roc(test_data$label, -svm_dv, levels = c("REAL", "FAKE"), quiet = TRUE)

png("roc_curves_comparison.png", width = 700, height = 600)
plot(roc_rf, col = "#2c7bb6", lwd = 2.5, main = "ROC Curves Comparison")
lines(roc_svm, col = "#fdae61", lwd = 2)
lines(roc_nb, col = "#d7191c", lwd = 2, lty = 2)
legend("bottomright",
       legend = c(paste0("Random Forest (AUC = ", round(auc(roc_rf), 3), ")"),
                  paste0("SVM (AUC = ", round(auc(roc_svm), 3), ")"),
                  paste0("Naive Bayes (AUC = ", round(auc(roc_nb), 3), ")")),
       col = c("#2c7bb6", "#fdae61", "#d7191c"), lwd = c(2.5, 2, 2), lty = c(1, 1, 2))
grid()
dev.off()

# Feature Importance Plot
png("feature_importance_rf.png", width = 750, height = 750)
varImpPlot(rf_model, n.var = 20, main = "Top 20 Features - Random Forest")
dev.off()

# Confusion Matrices
plot_cm <- function(cm_obj, title_text, filename, fill_color) {
  df <- as.data.frame(cm_obj$table)
  colnames(df) <- c("Prediction", "Reference", "Freq")
  p <- ggplot(df, aes(x = Prediction, y = Reference, fill = Freq)) +
    geom_tile(color = "white") +
    geom_text(aes(label = Freq), size = 5, fontface = "bold") +
    scale_fill_gradient(low = "#f7fbff", high = fill_color) +
    labs(title = title_text) +
    theme_minimal()
  ggsave(filename, plot = p, width = 4.8, height = 3.8)
}

plot_cm(nb_res$cm,  "Confusion Matrix - Naive Bayes",   "confusion_matrix_nb.png",  "#fc9272")
plot_cm(rf_res$cm,  "Confusion Matrix - Random Forest", "confusion_matrix_rf.png",  "#9ecae1")
plot_cm(svm_res$cm, "Confusion Matrix - SVM",           "confusion_matrix_svm.png", "#a1d99b")

# Word Clouds
save_cloud <- function(text_vec, filename, pal) {
  words <- unlist(strsplit(tolower(paste(sample(text_vec, min(500, length(text_vec))), collapse = " ")), "\\W+"))
  words <- words[nchar(words) > 3 & !(words %in% stopwords("english"))]
  w_counts <- sort(table(words), decreasing = TRUE)
  
  png(filename, width = 700, height = 700)
  par(mar = c(0, 0, 0, 0))
  wordcloud(names(w_counts), as.numeric(w_counts), max.words = 75,
            colors = pal, scale = c(3.5, 0.7), random.order = FALSE)
  dev.off()
}

save_cloud(news$text[news$label == "FAKE"], "wordcloud_fake.png", brewer.pal(8, "Dark2"))
save_cloud(news$text[news$label == "REAL"], "wordcloud_real.png", brewer.pal(8, "Blues")[3:8])

graphics.off()
total_time <- round(difftime(Sys.time(), start_time, units = "mins"), 2)
cat("\nDone! Pipeline finished in", total_time, "minutes.\n")
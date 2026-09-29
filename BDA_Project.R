library(shiny)
library(tm)
library(randomForest)
library(SnowballC)
library(httr)
library(jsonlite)

# Load trained model artifacts
rf_model <- readRDS("rf_model.rds")
features <- readRDS("features.rds")

# Google Fact Check Tools API Key
google_api_key <- "AIzaSyDuwC8qvg6qRZkMwLyjAB1VALm7gfZdbyY"

# Fact-check lookup function
fetch_fact_check <- function(query_text, api_key) {
  clean_query <- substr(query_text, 1, 100)
  url <- "https://factchecktools.googleapis.com/v1alpha1/claims:search"
  
  response <- tryCatch({
    GET(url, query = list(query = clean_query, key = api_key), timeout(5))
  }, error = function(e) NULL)
  
  if (!is.null(response) && http_status(response)$category == "Success") {
    content_parsed <- fromJSON(content(response, "text", encoding = "UTF-8"))
    if ("claims" %in% names(content_parsed) && length(content_parsed$claims) > 0) {
      claims <- content_parsed$claims
      reviews <- claims$claimReview[[1]]
      return(data.frame(
        Claim = claims$text[1],
        Verdict = reviews$textualRating[1],
        Publisher = reviews$publisher$name[1],
        URL = reviews$url[1],
        stringsAsFactors = FALSE
      ))
    }
  }
  return(NULL)
}

ui <- fluidPage(
  titlePanel("Global Fake News Detection System - BDA Live Demo"),
  sidebarLayout(
    sidebarPanel(
      helpText("Paste a news headline and article body below for automated machine learning classification and real-time fact-check verification."),
      textAreaInput("news_text", "Article Content:", rows = 8, 
                    placeholder = "Paste full news article text here..."),
      actionButton("predict_btn", "Analyze Article", class = "btn-primary"),
      hr(),
      helpText("Model: Random Forest (100 Trees | TF + Stylometrics)"),
      helpText("Ground Truth: Google Fact Check Tools API")
    ),
    mainPanel(
      h3("1. Model Classification Result"),
      uiOutput("prediction_box"),
      br(),
      h4("Model Confidence"),
      tableOutput("prob_table"),
      hr(),
      h3("2. External Fact-Check Benchmark (Ground Truth)"),
      uiOutput("fact_check_box")
    )
  )
)

server <- function(input, output) {
  
  observeEvent(input$predict_btn, {
    req(input$news_text)
    raw_text <- input$news_text
    
    # 1. Stylometric Feature Extraction on Uncleaned Text
    live_excl  <- length(regmatches(raw_text, gregexpr("!", raw_text))[[1]])
    live_caps  <- length(regmatches(raw_text, gregexpr("\\b[A-Z]{2,}\\b", raw_text))[[1]])
    live_words <- length(gregexpr("\\W+", raw_text)[[1]]) + 1
    
    # 2. Text Preprocessing & Cleaning
    clean_text <- gsub("\\(reuters\\)", "", raw_text, ignore.case = TRUE)
    clean_text <- gsub("reuters", "", clean_text, ignore.case = TRUE)
    clean_text <- gsub("^[A-Z ]+\\s*-\\s*", "", clean_text)
    
    corp <- VCorpus(VectorSource(clean_text))
    corp <- tm_map(corp, content_transformer(tolower))
    corp <- tm_map(corp, removePunctuation)
    corp <- tm_map(corp, removeNumbers)
    corp <- tm_map(corp, removeWords, stopwords("english"))
    corp <- tm_map(corp, stripWhitespace)
    corp <- tm_map(corp, stemDocument)
    
    # 3. Term Frequency Matrix Construction
    dtm_input <- DocumentTermMatrix(corp, control = list(weighting = weightTf))
    dtm_df <- as.data.frame(as.matrix(dtm_input))
    colnames(dtm_df) <- make.names(colnames(dtm_df))
    
    # 4. Align Feature Vector with Training Schema
    input_vector <- as.data.frame(matrix(0, nrow = 1, ncol = length(features)))
    colnames(input_vector) <- features
    
    common_terms <- intersect(colnames(input_vector), colnames(dtm_df))
    if (length(common_terms) > 0) {
      input_vector[1, common_terms] <- dtm_df[1, common_terms]
    }
    
    # Inject Stylometric Variables
    input_vector$excl_count <- live_excl
    input_vector$caps_count <- live_caps
    input_vector$word_count <- live_words
    
    # 5. Machine Learning Prediction
    prediction <- predict(rf_model, input_vector)
    probs <- predict(rf_model, input_vector, type = "prob")
    
    # 6. Render UI Outputs
    output$prediction_box <- renderUI({
      if (prediction == "FAKE") {
        div(style = "background-color: #f8d7da; color: #721c24; padding: 12px; border-radius: 5px; font-weight: bold; font-size: 18px;",
            "WARNING: This article is classified as FAKE NEWS")
      } else {
        div(style = "background-color: #d4edda; color: #155724; padding: 12px; border-radius: 5px; font-weight: bold; font-size: 18px;",
            "VERIFIED: This article is classified as REAL NEWS")
      }
    })
    
    output$prob_table <- renderTable({
      data.frame(
        Class = c("Fake News Probability", "Real News Probability"),
        Confidence = paste0(round(probs[1, ] * 100, 2), "%")
      )
    })
    
    # 7. Query Google Fact Check API
    fact_res <- fetch_fact_check(input$news_text, google_api_key)
    
    output$fact_check_box <- renderUI({
      if (!is.null(fact_res)) {
        div(style = "background-color: #e2e3e5; padding: 15px; border-radius: 5px;",
            tags$p(tags$b("Claim Reviewed: "), fact_res$Claim),
            tags$p(tags$b("Official Verdict: "), tags$span(style = "color: #b02a37; font-weight: bold;", fact_res$Verdict)),
            tags$p(tags$b("Verified By: "), fact_res$Publisher),
            tags$p(tags$b("Full Report: "), tags$a(href = fact_res$URL, target = "_blank", "Read verification article"))
        )
      } else {
        div(style = "background-color: #f8f9fa; padding: 15px; border-radius: 5px; color: #6c757d;",
            "No prior fact-checks indexed by Google for this exact phrasing. Model prediction relies on internal linguistic & stylistic evaluation.")
      }
    })
  })
}

shinyApp(ui = ui, server = server)
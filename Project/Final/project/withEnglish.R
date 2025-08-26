# Load required libraries
library(rvest)
library(dplyr)
library(readr)
library(purrr)
library(tm)
library(tokenizers)
library(SnowballC)
library(wordcloud)
library(RColorBrewer)
library(topicmodels)
library(tidytext)
library(ggplot2)
library(tidyr)

# ============================
# 1. Web Scraping Links from The Daily Star homepage
# ============================
base_url <- "https://www.thedailystar.net"
homepage <- read_html(base_url)

a_nodes <- homepage %>% html_nodes("a")
links_data <- data.frame(
  href = a_nodes %>% html_attr("href"),
  aria_label = a_nodes %>% html_attr("aria-label"),
  stringsAsFactors = FALSE
)

# Define patterns to keep - adapt based on actual article URL structure
article_patterns <- c("/news/", "/category/", "/opinion/", "/editorial/")  # example patterns

# Define social media domains to exclude
social_domains <- c("facebook.com", "twitter.com", "instagram.com", "linkedin.com", "youtube.com", "tiktok.com")

links_data <- links_data %>%
  filter(!is.na(href), !is.na(aria_label)) %>%
  mutate(href = ifelse(grepl("^http", href), href, paste0(base_url, href))) %>%
  filter(grepl(paste(article_patterns, collapse = "|"), href)) %>%
  filter(!grepl(paste(social_domains, collapse = "|"), href)) %>%
  distinct()

write_csv(links_data, "links_with_labels.csv")
cat("Saved", nrow(links_data), "links with aria-labels to links_with_labels.csv\n")

# ============================
# 2. Extract Article Texts from Links
# ============================
links_sample <- links_data$href

extract_article_details <- function(url) {
  tryCatch({
    page <- read_html(url)
    Sys.sleep(1) # polite pause
    
    panel_nodes <- page %>% html_nodes("div.tabs-panel.is-active")
    if (length(panel_nodes) > 0) {
      xml2::xml_remove(panel_nodes)
    }
    
    date_published <- page %>%
      html_node("meta[itemprop='datePublished'], meta[property='article:published_time']") %>%
      html_attr("content")
    if (is.null(date_published)) date_published <- NA
    
    paragraphs <- page %>%
      html_nodes("p") %>%
      html_text() %>%
      paste(collapse = " ")
    
    data.frame(
      url = url,
      article_text = paragraphs,
      date_published = date_published,
      stringsAsFactors = FALSE
    )
  }, error = function(e) {
    message("Failed to scrape: ", url)
    return(data.frame(
      url = url,
      article_text = NA,
      date_published = NA,
      stringsAsFactors = FALSE
    ))
  })
}

articles <- map_df(links_sample, extract_article_details)

write_csv(articles, "articles_with_text.csv")
cat("Saved scraped articles with text to articles_with_text.csv\n")

# ============================
# 3. Text Preprocessing
# ============================
data <- read_csv("articles_with_text.csv", show_col_types = FALSE)

corpus <- Corpus(VectorSource(data$article_text))

custom_stopwords <- c("said", "year", "will", "bangladesh", "s", "go", "told", "al", "jazeera",
                      "t", "sat", "hasn", "also", "many", "says", "percent")

corpus <- tm_map(corpus, content_transformer(tolower))
corpus <- tm_map(corpus, content_transformer(function(x) gsub("[^a-z ]", " ", x)))
corpus <- tm_map(corpus, removePunctuation)
corpus <- tm_map(corpus, removeNumbers)
corpus <- tm_map(corpus, removeWords, stopwords("english"))
corpus <- tm_map(corpus, removeWords, custom_stopwords)
corpus <- tm_map(corpus, stripWhitespace)
corpus <- tm_map(corpus, stemDocument, language = "english")  # optional stemming

# ============================
# 4. Exploratory Text Analysis
# ============================
tdm <- TermDocumentMatrix(corpus)
m <- as.matrix(tdm)
word_freqs <- sort(rowSums(m), decreasing = TRUE)
df_words <- data.frame(word = names(word_freqs), freq = word_freqs)

write_csv(df_words, "word_frequencies.csv")

set.seed(123)
png("wordcloud.png", width = 1000, height = 800)
wordcloud(words = df_words$word,
          freq = df_words$freq,
          min.freq = 3,
          max.words = 100,
          random.order = FALSE,
          colors = brewer.pal(8, "Dark2"))
dev.off()
cat("Word cloud saved to wordcloud.png\n")

top_words <- df_words[1:20, ]

png("top20_words_barplot.png", width = 1000, height = 600)
ggplot(top_words, aes(x = reorder(word, freq), y = freq)) +
  geom_bar(stat = "identity", fill = "steelblue") +
  coord_flip() +
  labs(title = "Top 20 Most Frequent Words",
       x = "Words",
       y = "Frequency") +
  theme_minimal()
dev.off()
cat("Top 20 words barplot saved to top20_words_barplot.png\n")

# ============================
# 5. Topic Modeling with LDA
# ============================
dtm <- DocumentTermMatrix(corpus)
row_totals <- apply(dtm, 1, sum)
dtm <- dtm[row_totals > 0, ]

num_topics <- 5
lda_model <- LDA(dtm, k = num_topics, control = list(seed = 42))

top_terms <- terms(lda_model, 10)
cat("Top 10 terms per topic:\n")
print(top_terms)

topic_probabilities <- tidy(lda_model, matrix = "gamma")

doc_topics <- topic_probabilities %>%
  filter(document %in% as.character(1:20)) %>%
  mutate(topic = factor(topic))

png("document_topic_distribution.png", width = 1000, height = 600)
ggplot(doc_topics, aes(x = document, y = gamma, fill = topic)) +
  geom_bar(stat = "identity", position = "stack") +
  labs(title = "Document-Topic Distribution (First 20 documents)",
       x = "Document",
       y = "Topic Probability") +
  theme_minimal()
dev.off()
cat("Document-topic distribution plot saved to document_topic_distribution.png\n")

topic_terms_df <- tidy(lda_model, matrix = "beta") %>%
  group_by(topic) %>%
  top_n(10, beta) %>%
  ungroup() %>%
  arrange(topic, -beta)

png("top_terms_per_topic.png", width = 1200, height = 800)
ggplot(topic_terms_df, aes(x = reorder_within(term, beta, topic), y = beta, fill = factor(topic))) +
  geom_col(show.legend = FALSE) +
  facet_wrap(~ topic, scales = "free") +
  scale_x_reordered() +
  coord_flip() +
  labs(title = "Top 10 Terms Per Topic",
       x = NULL,
       y = "Beta (Term Probability)") +
  theme_minimal()
dev.off()
cat("Top terms per topic plot saved to top_terms_per_topic.png\n")

# ============================
# 6. Save final cleaned text data and results
# ============================
cleaned_texts <- sapply(corpus, as.character)
final_data <- data.frame(url = data$url[1:length(cleaned_texts)],
                         cleaned_text = cleaned_texts,
                         stringsAsFactors = FALSE)

write_csv(final_data, "cleaned_articles_text.csv")

cat("Cleaned texts saved to cleaned_articles_text.csv\n")

# ================
# End of Project Workflow
# ================

cat("\nProject execution complete. Please check the generated CSV files and PNG plots for results.\n")

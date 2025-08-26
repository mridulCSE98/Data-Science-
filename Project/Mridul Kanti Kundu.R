# ============================
# Load Libraries
# ============================
library(rvest)
library(dplyr)
library(readr)
library(purrr)

library(tm)
library(stringr)
library(wordcloud)
library(RColorBrewer)
library(ggplot2)

library(topicmodels) 
library(tidytext)    
library(tidyr)
library(tokenizers)  # needed for tokenize_words()

# ============================
# Create Output Directory
# ============================
output_dir <- "tsports_data"
if (!dir.exists(output_dir)) {
  dir.create(output_dir)
}

# ============================
# 1. Web Scraping 
# ============================
base_url <- "https://www.tsports.com/"
homepage <- read_html(base_url)

a_nodes <- homepage %>% html_nodes("a")
links_data <- data.frame(
  href = a_nodes %>% html_attr("href"),
  stringsAsFactors = FALSE
)

article_patterns <- c("/news/", "/cricket/", "/football/", "/sports/", "/article/")
social_domains <- c("facebook.com", "twitter.com", "instagram.com", "linkedin.com", "youtube.com", "tiktok.com")

links_data <- links_data %>%
  filter(!is.na(href)) %>%
  mutate(href = ifelse(grepl("^http", href), href, paste0(base_url, sub("^/", "", href)))) %>%
  filter(grepl(paste(article_patterns, collapse = "|"), href, ignore.case = TRUE)) %>%
  filter(!grepl(paste(social_domains, collapse = "|"), href)) %>%
  distinct()

write_csv(links_data, file.path(output_dir, "article_links.csv"))
cat("Saved", nrow(links_data), "article links from tsports.com homepage\n")

# ============================
# 2. Extract Article Texts from Links
# ============================
links_sample <- links_data$href

extract_article_details <- function(url) {
  tryCatch({
    page <- read_html(url)
    Sys.sleep(1)
    
    paragraphs <- page %>%
      html_nodes("article p, .entry-content p, .post-content p") %>%
      html_text(trim = TRUE) %>%
      paste(collapse = " ")
    
    data.frame(
      url = url,
      article_text = paragraphs,
      stringsAsFactors = FALSE
    )
  }, error = function(e) {
    message("Failed to scrape: ", url)
    return(data.frame(
      url = url,
      article_text = NA,
      stringsAsFactors = FALSE
    ))
  })
}

articles <- map_df(links_sample, extract_article_details)

write_csv(articles, file.path(output_dir, "articles_with_text.csv"))
cat("Saved scraped articles with text to articles_with_text.csv\n")

# ============================
# 3. Text Preprocessing
# ============================
data <- read_csv(file.path(output_dir, "articles_with_text.csv"), show_col_types = FALSE)
texts <- data$article_text

# English Stopwords (custom set)
english_stopwords <- c(
  "a", "about", "above", "after", "again", "against", "all", "am", "an", "and", "any", "are",
  "aren't", "as", "at", "be", "because", "been", "before", "being", "below", "between", "both",
  "but", "by", "can't", "cannot", "could", "couldn't", "did", "didn't", "do", "does", "doesn't",
  "doing", "don't", "down", "during", "each", "few", "for", "from", "further", "had", "hadn't",
  "has", "hasn't", "have", "haven't", "having", "he", "he'd", "he'll", "he's", "her", "here",
  "here's", "hers", "herself", "him", "himself", "his", "how", "how's", "i", "i'd", "i'll", "i'm",
  "i've", "if", "in", "into", "is", "isn't", "it", "it's", "its", "itself", "let's", "me", "more",
  "most", "mustn't", "my", "myself", "no", "nor", "not", "of", "off", "on", "once", "only", "or",
  "other", "ought", "our", "ours", "ourselves", "out", "over", "own", "same", "shan't", "she",
  "she'd", "she'll", "she's", "should", "shouldn't", "so", "some", "such", "than", "that", "that's",
  "the", "their", "theirs", "them", "themselves", "then", "there", "there's", "these", "they",
  "they'd", "they'll", "they're", "they've", "this", "those", "through", "to", "too", "under",
  "until", "up", "very", "was", "wasn't", "we", "we'd", "we'll", "we're", "we've", "were", "weren't",
  "what", "what's", "when", "when's", "where", "where's", "which", "while", "who", "who's", "whom",
  "why", "why's", "with", "won't", "would", "wouldn't", "you", "you'd", "you'll", "you're", "you've",
  "your", "yours", "yourself", "yourselves"
)

clean_text <- function(text) {
  text %>%
    str_to_lower() %>%
    str_replace_all("[^a-z\\s]", " ") %>%
    str_replace_all("[[:punct:]]", " ") %>%
    str_replace_all("[0-9]", " ") %>%
    str_replace_all("\\b\\w{1,3}\\b", " ") %>%
    str_squish()
}

cleaned_texts <- sapply(texts, clean_text, USE.NAMES = FALSE)

remove_stopwords <- function(text) {
  words <- unlist(str_split(text, "\\s+"))
  filtered <- words[!words %in% english_stopwords]
  paste(filtered, collapse = " ")
}

final_texts <- sapply(cleaned_texts, remove_stopwords, USE.NAMES = FALSE)
corpus <- Corpus(VectorSource(final_texts))

# Tokenization 
tokenized_texts <- lapply(final_texts, function(text) {
  unlist(tokenize_words(text))
})

cat("Tokens of document 1:\n")
print(tokenized_texts[[1]])

# ============================
# 4. Exploratory Text Analysis
# ============================
tdm <- TermDocumentMatrix(corpus)
m <- as.matrix(tdm)

word_freqs <- sort(rowSums(m), decreasing = TRUE)
df_words <- data.frame(word = names(word_freqs), freq = word_freqs)

write_csv(df_words, file.path(output_dir, "word_frequencies.csv"))

set.seed(123)
png(file.path(output_dir, "wordcloud.png"), width = 1000, height = 800)
wordcloud(words = df_words$word,
          freq = df_words$freq,
          min.freq = 3,
          max.words = 100,
          random.order = FALSE,
          colors = brewer.pal(8, "Dark2"))
dev.off()
cat("Word cloud saved to wordcloud.png\n")

top_words <- df_words[1:20, ]

png(file.path(output_dir, "top20_words_barplot.png"), width = 1000, height = 600)
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

png(file.path(output_dir, "document_topic_distribution.png"), width = 1000, height = 600)
ggplot(doc_topics, aes(x = document, y = gamma, fill = topic)) +
  geom_bar(stat = "identity", position = "stack") +
  labs(title = "Document-Topic Distribution",
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

png(file.path(output_dir, "top_terms_per_topic.png"), width = 1200, height = 800)
ggplot(topic_terms_df, aes(x = reorder_within(term, beta, topic), y = beta, fill = factor(topic))) +
  geom_col(show.legend = FALSE) +
  facet_wrap(~ topic, scales = "free") +
  scale_x_reordered() +
  coord_flip() +
  labs(title = "Top 10 Terms Per Topic",
       x = NULL,
       y = "Term Probability") +
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

write_csv(final_data, file.path(output_dir, "cleaned_articles_text.csv"))

cat("Cleaned texts saved to cleaned_articles_text.csv\n")

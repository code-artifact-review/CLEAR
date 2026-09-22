suppressPackageStartupMessages({
  library(ggplot2)
  library(scales)
  library(jsonlite)
})

DATA_ROOT <- file.path(
  "..",
  "Data"
)

DATASET_NAME <- "linedp_dataset"

CLEAR_AGGREGATION_METHODS <- c(
  "sum",
  "max",
  "top3avg",
  "top2avg"
)

CLEAR_AGGREGATION_METHOD <- NULL

TOKEN_SCORE_ROOT <- file.path(
  DATA_ROOT,
  DATASET_NAME,
  "token_scores"
)

OUTPUT_ROOT <- NULL
DISTRIBUTION_FIGURE_PATH <- NULL
PAIRED_DIFFERENCE_FIGURE_PATH <- NULL
PAIRED_DIFFERENCE_RESULT_PATH <- NULL
PAIRWISE_RESULT_PATH <- NULL
SUMMARY_RESULT_PATH <- NULL
PAIRED_COMPARISON_PATH <- NULL
ZERO_TEST_RESULT_PATH <- NULL

METHOD_ORDER <- c(
  "without_confidence",
  "with_confidence"
)

METHOD_LABELS <- NULL

BLUE_COLOR <- "#547DA3"
ORANGE_COLOR <- "#DA936A"
GREY_COLOR <- "#A9A9A9"

METHOD_COLORS <- c(
  "without_confidence" =
    BLUE_COLOR,
  "with_confidence" =
    ORANGE_COLOR
)

METHOD_SHAPES <- c(
  "without_confidence" = 16,
  "with_confidence" = 18
)

METHOD_SCORE_FILENAMES <- c(
  "without_confidence" =
    "{release}_suspiciousness_token_scores.json",
  "with_confidence" =
    "{release}_confidence_enhanced_token_scores.json"
)

DISTRIBUTION_OUTPUT_WIDTH <- 10
DISTRIBUTION_OUTPUT_HEIGHT <- 8
PAIRED_DIFFERENCE_OUTPUT_WIDTH <- 12
PAIRED_DIFFERENCE_OUTPUT_HEIGHT <- 7.32
OUTPUT_DPI <- 300

SUMMARY_Y_LIMITS <- c(
  0.10,
  0.65
)

SUMMARY_Y_BREAK_BY <- 0.10

# Use NULL for a data-driven paired-difference axis.
# Set a fixed range such as c(0.000, 0.004) when needed.
PAIRED_DIFFERENCE_X_LIMITS <- NULL

PAIRED_DIFFERENCE_X_ACCURACY <- 0.0005

LINEDP_PROJECTS <- list(
  "activemq" = c(
    "activemq-5.0.0",
    "activemq-5.1.0",
    "activemq-5.2.0",
    "activemq-5.3.0",
    "activemq-5.8.0"
  ),
  "camel" = c(
    "camel-1.4.0",
    "camel-2.9.0",
    "camel-2.10.0",
    "camel-2.11.0"
  ),
  "derby" = c(
    "derby-10.2.1.6",
    "derby-10.3.1.4",
    "derby-10.5.1.1"
  ),
  "groovy" = c(
    "groovy-1_5_7",
    "groovy-1_6_BETA_1",
    "groovy-1_6_BETA_2"
  ),
  "hbase" = c(
    "hbase-0.94.0",
    "hbase-0.95.0",
    "hbase-0.95.2"
  ),
  "hive" = c(
    "hive-0.9.0",
    "hive-0.10.0",
    "hive-0.12.0"
  ),
  "jruby" = c(
    "jruby-1.1",
    "jruby-1.4.0",
    "jruby-1.5.0",
    "jruby-1.7.0.preview1"
  ),
  "lucene" = c(
    "lucene-2.3.0",
    "lucene-2.9.0",
    "lucene-3.0.0",
    "lucene-3.1"
  ),
  "wicket" = c(
    "wicket-1.3.0-incubating-beta-1",
    "wicket-1.3.0-beta2",
    "wicket-1.5.3"
  )
)


build_release_pairs <- function(
    projects
)
{
  pair_rows <- list()
  row_index <- 1
  pair_order <- 1
  
  for (project_name in names(projects)) {
    releases <- projects[[
      project_name
    ]]
    
    if (length(releases) < 2) {
      next
    }
    
    for (
      release_index in
      seq_len(
        length(releases) - 1
      )
    ) {
      pair_rows[[
        row_index
      ]] <- data.frame(
        project =
          project_name,
        source_release =
          releases[
            release_index
          ],
        target_release =
          releases[
            release_index + 1
          ],
        pair_id = paste0(
          releases[
            release_index
          ],
          "_to_",
          releases[
            release_index + 1
          ]
        ),
        pair_order =
          pair_order,
        stringsAsFactors = FALSE
      )
      
      row_index <- row_index + 1
      pair_order <- pair_order + 1
    }
  }
  
  if (length(pair_rows) == 0) {
    stop(
      "No adjacent release pairs were configured."
    )
  }
  
  return(
    do.call(
      rbind,
      pair_rows
    )
  )
}


get_token_score_path <- function(
    release,
    method
)
{
  if (!(method %in% names(
    METHOD_SCORE_FILENAMES
  ))) {
    stop(
      paste0(
        "Unsupported token-score method: ",
        method
      )
    )
  }
  
  filename <- gsub(
    "\\{release\\}",
    release,
    METHOD_SCORE_FILENAMES[[
      method
    ]]
  )
  
  return(
    file.path(
      TOKEN_SCORE_ROOT,
      filename
    )
  )
}


load_token_scores <- function(
    release,
    method
)
{
  input_path <- get_token_score_path(
    release,
    method
  )
  
  if (!file.exists(input_path)) {
    stop(
      paste0(
        "Token-score file does not exist: ",
        input_path
      )
    )
  }
  
  loaded_data <- fromJSON(
    input_path,
    simplifyVector = TRUE
  )
  
  raw_scores <- unlist(
    loaded_data,
    use.names = TRUE
  )
  
  if (
    length(
      raw_scores
    ) == 0 ||
    is.null(
      names(
        raw_scores
      )
    )
  ) {
    stop(
      paste0(
        "Expected a non-empty named token-score object in ",
        input_path,
        "."
      )
    )
  }
  
  token_names <- names(
    raw_scores
  )
  
  token_scores <- suppressWarnings(
    as.numeric(
      raw_scores
    )
  )
  
  names(
    token_scores
  ) <- token_names
  
  finite_mask <- is.finite(
    token_scores
  )
  
  token_scores <- token_scores[
    finite_mask
  ]
  
  if (length(token_scores) == 0) {
    stop(
      paste0(
        "No finite token scores were found in ",
        input_path,
        "."
      )
    )
  }
  
  return(
    token_scores
  )
}


calculate_spearman <- function(
    source_scores,
    target_scores
)
{
  valid_mask <- (
    is.finite(
      source_scores
    ) &
      is.finite(
        target_scores
      )
  )
  
  source_scores <- source_scores[
    valid_mask
  ]
  
  target_scores <- target_scores[
    valid_mask
  ]
  
  if (
    length(
      source_scores
    ) < 3 ||
    length(
      unique(
        source_scores
      )
    ) < 2 ||
    length(
      unique(
        target_scores
      )
    ) < 2
  ) {
    return(
      list(
        correlation = NA_real_,
        p_value = NA_real_,
        valid_token_count =
          length(
            source_scores
          )
      )
    )
  }
  
  correlation_value <- suppressWarnings(
    cor(
      source_scores,
      target_scores,
      method = "spearman"
    )
  )
  
  correlation_test <- suppressWarnings(
    cor.test(
      source_scores,
      target_scores,
      method = "spearman",
      exact = FALSE
    )
  )
  
  return(
    list(
      correlation =
        unname(
          correlation_value
        ),
      p_value =
        correlation_test$p.value,
      valid_token_count =
        length(
          source_scores
        )
    )
  )
}


build_empty_pair_rows <- function(
    pair_row,
    common_token_count
)
{
  return(
    data.frame(
      project = rep(
        pair_row$project,
        length(
          METHOD_ORDER
        )
      ),
      source_release = rep(
        pair_row$source_release,
        length(
          METHOD_ORDER
        )
      ),
      target_release = rep(
        pair_row$target_release,
        length(
          METHOD_ORDER
        )
      ),
      pair_id = rep(
        pair_row$pair_id,
        length(
          METHOD_ORDER
        )
      ),
      pair_order = rep(
        pair_row$pair_order,
        length(
          METHOD_ORDER
        )
      ),
      method =
        METHOD_ORDER,
      common_token_count =
        common_token_count,
      valid_token_count =
        0,
      source_nonzero_token_count =
        0,
      target_nonzero_token_count =
        0,
      both_nonzero_token_count =
        0,
      spearman_correlation =
        NA_real_,
      spearman_p_value =
        NA_real_,
      stringsAsFactors = FALSE
    )
  )
}


analyze_one_release_pair <- function(
    pair_row
)
{
  source_without <- load_token_scores(
    pair_row$source_release,
    "without_confidence"
  )
  
  target_without <- load_token_scores(
    pair_row$target_release,
    "without_confidence"
  )
  
  source_with <- load_token_scores(
    pair_row$source_release,
    "with_confidence"
  )
  
  target_with <- load_token_scores(
    pair_row$target_release,
    "with_confidence"
  )
  
  common_tokens <- sort(
    Reduce(
      intersect,
      list(
        names(
          source_without
        ),
        names(
          target_without
        ),
        names(
          source_with
        ),
        names(
          target_with
        )
      )
    )
  )
  
  common_token_count <- length(
    common_tokens
  )
  
  if (common_token_count < 3) {
    warning(
      paste0(
        "[",
        pair_row$pair_id,
        "] Fewer than three common tokens were found across the two releases and two scoring methods."
      )
    )
    
    return(
      build_empty_pair_rows(
        pair_row,
        common_token_count
      )
    )
  }
  
  result_rows <- list()
  
  for (
    method_index in
    seq_along(
      METHOD_ORDER
    )
  ) {
    method <- METHOD_ORDER[
      method_index
    ]
    
    if (method == "without_confidence") {
      source_scores <- source_without[
        common_tokens
      ]
      
      target_scores <- target_without[
        common_tokens
      ]
    } else {
      source_scores <- source_with[
        common_tokens
      ]
      
      target_scores <- target_with[
        common_tokens
      ]
    }
    
    spearman_result <- calculate_spearman(
      source_scores,
      target_scores
    )
    
    result_rows[[
      method_index
    ]] <- data.frame(
      project =
        pair_row$project,
      source_release =
        pair_row$source_release,
      target_release =
        pair_row$target_release,
      pair_id =
        pair_row$pair_id,
      pair_order =
        pair_row$pair_order,
      method =
        method,
      common_token_count =
        common_token_count,
      valid_token_count =
        spearman_result$valid_token_count,
      source_nonzero_token_count =
        sum(
          source_scores != 0
        ),
      target_nonzero_token_count =
        sum(
          target_scores != 0
        ),
      both_nonzero_token_count =
        sum(
          source_scores != 0 &
            target_scores != 0
        ),
      spearman_correlation =
        spearman_result$correlation,
      spearman_p_value =
        spearman_result$p_value,
      stringsAsFactors = FALSE
    )
  }
  
  return(
    do.call(
      rbind,
      result_rows
    )
  )
}


analyze_all_release_pairs <- function(
    release_pairs
)
{
  result_rows <- list()
  
  for (
    pair_index in
    seq_len(
      nrow(
        release_pairs
      )
    )
  ) {
    pair_row <- release_pairs[
      pair_index,
      ,
      drop = FALSE
    ]
    
    message(
      paste0(
        "[",
        pair_row$pair_id,
        "] Calculating Spearman correlation on adjacent-release common tokens."
      )
    )
    
    result_rows[[
      pair_index
    ]] <- analyze_one_release_pair(
      pair_row
    )
  }
  
  result_data <- do.call(
    rbind,
    result_rows
  )
  
  result_data$method <- factor(
    result_data$method,
    levels = METHOD_ORDER
  )
  
  return(
    result_data
  )
}


calculate_t_confidence_interval <- function(
    values,
    confidence_level = 0.95
)
{
  finite_values <- values[
    is.finite(
      values
    )
  ]
  
  sample_count <- length(
    finite_values
  )
  
  if (sample_count == 0) {
    return(
      c(
        NA_real_,
        NA_real_
      )
    )
  }
  
  if (sample_count == 1) {
    return(
      c(
        finite_values[1],
        finite_values[1]
      )
    )
  }
  
  standard_error <- (
    sd(
      finite_values
    ) /
      sqrt(
        sample_count
      )
  )
  
  alpha <- (
    1 -
      confidence_level
  )
  
  critical_value <- qt(
    1 -
      alpha /
      2,
    df =
      sample_count -
      1
  )
  
  mean_value <- mean(
    finite_values
  )
  
  return(
    c(
      mean_value -
        critical_value *
        standard_error,
      mean_value +
        critical_value *
        standard_error
    )
  )
}


calculate_rank_biserial <- function(
    differences
)
{
  finite_differences <- differences[
    is.finite(
      differences
    ) &
      differences != 0
  ]
  
  if (length(finite_differences) == 0) {
    return(0)
  }
  
  absolute_ranks <- rank(
    abs(
      finite_differences
    ),
    ties.method = "average"
  )
  
  positive_rank_sum <- sum(
    absolute_ranks[
      finite_differences > 0
    ]
  )
  
  negative_rank_sum <- sum(
    absolute_ranks[
      finite_differences < 0
    ]
  )
  
  denominator <- (
    positive_rank_sum +
      negative_rank_sum
  )
  
  if (denominator == 0) {
    return(0)
  }
  
  return(
    (
      positive_rank_sum -
        negative_rank_sum
    ) /
      denominator
  )
}


summarize_spearman <- function(
    result_data
)
{
  summary_rows <- list()
  
  for (
    method_index in
    seq_along(
      METHOD_ORDER
    )
  ) {
    method <- METHOD_ORDER[
      method_index
    ]
    
    values <- result_data$spearman_correlation[
      as.character(
        result_data$method
      ) == method
    ]
    
    values <- values[
      is.finite(
        values
      )
    ]
    
    if (length(values) == 0) {
      stop(
        paste0(
          "No finite Spearman correlations were found for ",
          method,
          "."
        )
      )
    }
    
    confidence_interval <- (
      calculate_t_confidence_interval(
        values
      )
    )
    
    summary_rows[[
      method_index
    ]] <- data.frame(
      method =
        method,
      method_label =
        METHOD_LABELS[[
          method
        ]],
      release_pair_count =
        length(
          values
        ),
      mean_correlation =
        mean(
          values
        ),
      median_correlation =
        median(
          values
        ),
      variance =
        if (
          length(
            values
          ) > 1
        ) {
          var(
            values
          )
        } else {
          0
        },
      standard_deviation =
        if (
          length(
            values
          ) > 1
        ) {
          sd(
            values
          )
        } else {
          0
        },
      ci_lower =
        confidence_interval[1],
      ci_upper =
        confidence_interval[2],
      positive_pair_count =
        sum(
          values > 0
        ),
      zero_pair_count =
        sum(
          values == 0
        ),
      negative_pair_count =
        sum(
          values < 0
        ),
      stringsAsFactors = FALSE
    )
  }
  
  summary_data <- do.call(
    rbind,
    summary_rows
  )
  
  summary_data$method <- factor(
    summary_data$method,
    levels = METHOD_ORDER
  )
  
  return(
    summary_data
  )
}


calculate_paired_comparison <- function(
    result_data
)
{
  without_data <- result_data[
    as.character(
      result_data$method
    ) == "without_confidence",
    c(
      "pair_id",
      "spearman_correlation"
    ),
    drop = FALSE
  ]
  
  with_data <- result_data[
    as.character(
      result_data$method
    ) == "with_confidence",
    c(
      "pair_id",
      "spearman_correlation"
    ),
    drop = FALSE
  ]
  
  names(
    without_data
  )[2] <- "without_confidence_correlation"
  
  names(
    with_data
  )[2] <- "with_confidence_correlation"
  
  paired_data <- merge(
    without_data,
    with_data,
    by = "pair_id",
    all = FALSE
  )
  
  paired_data <- paired_data[
    is.finite(
      paired_data$without_confidence_correlation
    ) &
      is.finite(
        paired_data$with_confidence_correlation
      ),
    ,
    drop = FALSE
  ]
  
  paired_difference <- (
    paired_data$with_confidence_correlation -
      paired_data$without_confidence_correlation
  )
  
  difference_confidence_interval <- (
    calculate_t_confidence_interval(
      paired_difference
    )
  )
  
  wilcoxon_result <- suppressWarnings(
    wilcox.test(
      paired_difference,
      mu = 0,
      alternative = "two.sided",
      exact = FALSE
    )
  )
  
  return(
    data.frame(
      release_pair_count =
        nrow(
          paired_data
        ),
      without_confidence_mean =
        mean(
          paired_data$without_confidence_correlation
        ),
      with_confidence_mean =
        mean(
          paired_data$with_confidence_correlation
        ),
      mean_paired_difference =
        mean(
          paired_difference
        ),
      difference_ci_lower =
        difference_confidence_interval[1],
      difference_ci_upper =
        difference_confidence_interval[2],
      median_paired_difference =
        median(
          paired_difference
        ),
      improved_pair_count =
        sum(
          paired_difference > 0
        ),
      tied_pair_count =
        sum(
          paired_difference == 0
        ),
      degraded_pair_count =
        sum(
          paired_difference < 0
        ),
      paired_rank_biserial =
        calculate_rank_biserial(
          paired_difference
        ),
      wilcoxon_p_value =
        wilcoxon_result$p.value,
      stringsAsFactors = FALSE
    )
  )
}


calculate_zero_tests <- function(
    result_data
)
{
  test_rows <- list()
  
  for (
    method_index in
    seq_along(
      METHOD_ORDER
    )
  ) {
    method <- METHOD_ORDER[
      method_index
    ]
    
    values <- result_data$spearman_correlation[
      as.character(
        result_data$method
      ) == method
    ]
    
    values <- values[
      is.finite(
        values
      )
    ]
    
    two_sided_result <- suppressWarnings(
      wilcox.test(
        values,
        mu = 0,
        alternative = "two.sided",
        exact = FALSE
      )
    )
    
    greater_result <- suppressWarnings(
      wilcox.test(
        values,
        mu = 0,
        alternative = "greater",
        exact = FALSE
      )
    )
    
    test_rows[[
      method_index
    ]] <- data.frame(
      method =
        method,
      method_label =
        METHOD_LABELS[[
          method
        ]],
      release_pair_count =
        length(
          values
        ),
      mean_correlation =
        mean(
          values
        ),
      median_correlation =
        median(
          values
        ),
      positive_pair_count =
        sum(
          values > 0
        ),
      zero_pair_count =
        sum(
          values == 0
        ),
      negative_pair_count =
        sum(
          values < 0
        ),
      rank_biserial_vs_zero =
        calculate_rank_biserial(
          values
        ),
      two_sided_p_value =
        two_sided_result$p.value,
      greater_than_zero_p_value =
        greater_result$p.value,
      stringsAsFactors = FALSE
    )
  }
  
  test_data <- do.call(
    rbind,
    test_rows
  )
  
  test_data$holm_two_sided_p_value <- p.adjust(
    test_data$two_sided_p_value,
    method = "holm"
  )
  
  test_data$holm_greater_than_zero_p_value <- p.adjust(
    test_data$greater_than_zero_p_value,
    method = "holm"
  )
  
  return(
    test_data
  )
}


format_p_value <- function(
    p_value
)
{
  if (!is.finite(p_value)) {
    return("NA")
  }
  
  if (p_value < 0.001) {
    return("< 0.001")
  }
  
  return(
    formatC(
      p_value,
      format = "f",
      digits = 3
    )
  )
}


prepare_paired_difference_data <- function(
    result_data
)
{
  without_data <- result_data[
    as.character(
      result_data$method
    ) == "without_confidence",
    c(
      "project",
      "source_release",
      "target_release",
      "pair_id",
      "pair_order",
      "spearman_correlation"
    ),
    drop = FALSE
  ]
  
  with_data <- result_data[
    as.character(
      result_data$method
    ) == "with_confidence",
    c(
      "pair_id",
      "spearman_correlation"
    ),
    drop = FALSE
  ]
  
  names(
    without_data
  )[6] <- "without_confidence_correlation"
  
  names(
    with_data
  )[2] <- "with_confidence_correlation"
  
  paired_data <- merge(
    without_data,
    with_data,
    by = "pair_id",
    all = FALSE
  )
  
  paired_data <- paired_data[
    is.finite(
      paired_data$without_confidence_correlation
    ) &
      is.finite(
        paired_data$with_confidence_correlation
      ),
    ,
    drop = FALSE
  ]
  
  paired_data$paired_difference <- (
    paired_data$with_confidence_correlation -
      paired_data$without_confidence_correlation
  )
  
  source_release_labels <- mapply(
    function(
    project,
    release
    )
    {
      return(
        sub(
          paste0(
            "^",
            project,
            "-"
          ),
          "",
          release
        )
      )
    },
    paired_data$project,
    paired_data$source_release,
    USE.NAMES = FALSE
  )
  
  target_release_labels <- mapply(
    function(
    project,
    release
    )
    {
      return(
        sub(
          paste0(
            "^",
            project,
            "-"
          ),
          "",
          release
        )
      )
    },
    paired_data$project,
    paired_data$target_release,
    USE.NAMES = FALSE
  )
  
  paired_data$pair_label <- paste0(
    paired_data$project,
    ": ",
    source_release_labels,
    " \u2192 ",
    target_release_labels
  )
  
  paired_data <- paired_data[
    order(
      paired_data$paired_difference,
      paired_data$pair_order
    ),
    ,
    drop = FALSE
  ]
  
  paired_data$pair_label <- factor(
    paired_data$pair_label,
    levels =
      paired_data$pair_label
  )
  
  return(
    paired_data
  )
}


calculate_difference_axis_limits <- function(
    paired_difference_data
)
{
  if (!is.null(PAIRED_DIFFERENCE_X_LIMITS)) {
    if (
      length(
        PAIRED_DIFFERENCE_X_LIMITS
      ) != 2 ||
      any(
        !is.finite(
          PAIRED_DIFFERENCE_X_LIMITS
        )
      ) ||
      PAIRED_DIFFERENCE_X_LIMITS[1] >=
      PAIRED_DIFFERENCE_X_LIMITS[2]
    ) {
      stop(
        "PAIRED_DIFFERENCE_X_LIMITS must contain two increasing finite values."
      )
    }
    
    return(
      PAIRED_DIFFERENCE_X_LIMITS
    )
  }
  
  difference_values <- (
    paired_difference_data$paired_difference
  )
  
  difference_values <- difference_values[
    is.finite(
      difference_values
    )
  ]
  
  if (length(difference_values) == 0) {
    stop(
      "No finite paired differences were found."
    )
  }
  
  observed_range <- diff(
    range(
      difference_values
    )
  )
  
  if (
    !is.finite(
      observed_range
    ) ||
    observed_range <= 0
  ) {
    observed_range <- max(
      abs(
        difference_values
      ),
      PAIRED_DIFFERENCE_X_ACCURACY
    )
  }
  
  padding_value <- max(
    observed_range *
      0.15,
    PAIRED_DIFFERENCE_X_ACCURACY
  )
  
  return(
    c(
      min(
        0,
        min(
          difference_values
        ) -
          padding_value
      ),
      max(
        0,
        max(
          difference_values
        ) +
          padding_value
      )
    )
  )
}


build_distribution_plot <- function(
    result_data,
    summary_data,
    paired_comparison,
    zero_test_data
)
{
  plot_data <- result_data[
    is.finite(
      result_data$spearman_correlation
    ),
    ,
    drop = FALSE
  ]
  
  plot_data$method <- factor(
    plot_data$method,
    levels = METHOD_ORDER
  )
  
  summary_data$method <- factor(
    summary_data$method,
    levels = METHOD_ORDER
  )
  
  without_zero_test <- zero_test_data[
    zero_test_data$method ==
      "without_confidence",
    ,
    drop = FALSE
  ]
  
  with_zero_test <- zero_test_data[
    zero_test_data$method ==
      "with_confidence",
    ,
    drop = FALSE
  ]
  
  subtitle_text <- paste0(
    "Positive-correlation test: CLEAR w/o Confidence p ",
    format_p_value(
      without_zero_test$holm_greater_than_zero_p_value
    ),
    "; CLEAR p ",
    format_p_value(
      with_zero_test$holm_greater_than_zero_p_value
    ),
    ".\nPaired method comparison: \u0394 = ",
    number(
      paired_comparison$mean_paired_difference,
      accuracy = 0.001
    ),
    " (95% CI ",
    number(
      paired_comparison$difference_ci_lower,
      accuracy = 0.001
    ),
    ", ",
    number(
      paired_comparison$difference_ci_upper,
      accuracy = 0.001
    ),
    "), Wilcoxon p ",
    format_p_value(
      paired_comparison$wilcoxon_p_value
    ),
    "."
  )
  
  return(
    ggplot(
      plot_data,
      aes(
        x =
          method,
        y =
          spearman_correlation,
        group =
          pair_id
      )
    ) +
      geom_boxplot(
        aes(
          fill =
            method,
          group =
            method
        ),
        width = 0.36,
        alpha = 0.15,
        outlier.shape = NA,
        color = "#666666",
        linewidth = 0.50,
        show.legend = FALSE
      ) +
      geom_line(
        color =
          GREY_COLOR,
        linewidth = 0.50,
        alpha = 0.55
      ) +
      geom_point(
        aes(
          color =
            method,
          shape =
            method
        ),
        size = 3.2,
        alpha = 0.85
      ) +
      geom_errorbar(
        data =
          summary_data,
        aes(
          x =
            method,
          ymin =
            ci_lower,
          ymax =
            ci_upper,
          group = NULL
        ),
        inherit.aes = FALSE,
        width = 0.10,
        linewidth = 0.90,
        color = "black"
      ) +
      geom_point(
        data =
          summary_data,
        aes(
          x =
            method,
          y =
            mean_correlation,
          fill =
            method
        ),
        inherit.aes = FALSE,
        shape = 21,
        size = 5.0,
        stroke = 0.95,
        color = "black"
      ) +
      scale_color_manual(
        name = NULL,
        values =
          METHOD_COLORS,
        breaks =
          METHOD_ORDER,
        labels =
          METHOD_LABELS[
            METHOD_ORDER
          ],
        drop = FALSE
      ) +
      scale_fill_manual(
        name = NULL,
        values =
          METHOD_COLORS,
        breaks =
          METHOD_ORDER,
        labels =
          METHOD_LABELS[
            METHOD_ORDER
          ],
        drop = FALSE
      ) +
      scale_shape_manual(
        name = NULL,
        values =
          METHOD_SHAPES,
        breaks =
          METHOD_ORDER,
        labels =
          METHOD_LABELS[
            METHOD_ORDER
          ],
        drop = FALSE
      ) +
      scale_x_discrete(
        labels =
          METHOD_LABELS[
            METHOD_ORDER
          ],
        drop = FALSE
      ) +
      scale_y_continuous(
        limits =
          SUMMARY_Y_LIMITS,
        breaks = seq(
          SUMMARY_Y_LIMITS[1],
          SUMMARY_Y_LIMITS[2],
          by =
            SUMMARY_Y_BREAK_BY
        ),
        labels = label_number(
          accuracy = 0.01
        ),
        expand = expansion(
          mult = c(
            0.02,
            0.04
          )
        )
      ) +
      guides(
        color = "none",
        fill = "none",
        shape = "none"
      ) +
      labs(
        x = NULL,
        y =
          "Adjacent-Release Spearman Rank Correlation",
        title =
          "Distribution of Adjacent-Release Token-Score Correlations",
        subtitle =
          subtitle_text,
        caption =
          "Small points represent release pairs; black-edged large points and error bars represent means and 95% t-based \nconfidence intervals."
      ) +
      theme_minimal() +
      theme(
        text = element_text(
          family = "Arial"
        ),
        plot.background = element_rect(
          fill = "white",
          color = NA
        ),
        panel.background = element_rect(
          fill = "white",
          color = NA
        ),
        panel.grid.major.x =
          element_blank(),
        panel.grid.major.y =
          element_line(
            color = "#D7D7D7",
            linewidth = 0.50
          ),
        panel.grid.minor =
          element_blank(),
        plot.title =
          element_text(
            size = 21,
            face = "bold",
            color = "black",
            margin = margin(
              b = 7
            )
          ),
        plot.subtitle =
          element_text(
            size = 12.5,
            color = "#2F2F2F",
            lineheight = 1.20,
            margin = margin(
              b = 12
            )
          ),
        axis.title.y =
          element_text(
            size = 18,
            color = "black",
            margin = margin(
              r = 12
            )
          ),
        axis.text.x =
          element_text(
            size = 16,
            color = "black",
            margin = margin(
              t = 10
            )
          ),
        axis.text.y =
          element_text(
            size = 14,
            color = "black",
            margin = margin(
              r = 7
            )
          ),
        plot.caption =
          element_text(
            size = 11.5,
            color = "#333333",
            hjust = 0,
            margin = margin(
              t = 12
            )
          ),
        plot.margin = margin(
          t = 20,
          r = 24,
          b = 18,
          l = 24
        )
      )
  )
}


build_paired_difference_plot <- function(
    paired_difference_data,
    paired_comparison
)
{
  difference_limits <- calculate_difference_axis_limits(
    paired_difference_data
  )
  
  difference_breaks <- pretty(
    difference_limits,
    n = 6
  )
  
  difference_breaks <- difference_breaks[
    difference_breaks >=
      difference_limits[1] &
      difference_breaks <=
      difference_limits[2]
  ]
  
  subtitle_text <- paste0(
    "Mean paired \u0394\u03c1 = ",
    number(
      paired_comparison$mean_paired_difference,
      accuracy = 0.0001
    ),
    " (95% CI ",
    number(
      paired_comparison$difference_ci_lower,
      accuracy = 0.0001
    ),
    ", ",
    number(
      paired_comparison$difference_ci_upper,
      accuracy = 0.0001
    ),
    "); improved/tied/degraded = ",
    paired_comparison$improved_pair_count,
    "/",
    paired_comparison$tied_pair_count,
    "/",
    paired_comparison$degraded_pair_count,
    "; rank-biserial = ",
    number(
      paired_comparison$paired_rank_biserial,
      accuracy = 0.01
    ),
    "."
  )
  
  return(
    ggplot(
      paired_difference_data,
      aes(
        x =
          paired_difference,
        y =
          pair_label
      )
    ) +
      annotate(
        "rect",
        xmin =
          paired_comparison$difference_ci_lower,
        xmax =
          paired_comparison$difference_ci_upper,
        ymin = -Inf,
        ymax = Inf,
        fill =
          ORANGE_COLOR,
        alpha = 0.10
      ) +
      geom_vline(
        xintercept = 0,
        color = "#969696",
        linewidth = 0.65,
        linetype = "dashed"
      ) +
      geom_vline(
        xintercept =
          paired_comparison$mean_paired_difference,
        color =
          ORANGE_COLOR,
        linewidth = 0.85
      ) +
      geom_segment(
        aes(
          x = 0,
          xend =
            paired_difference,
          y =
            pair_label,
          yend =
            pair_label
        ),
        color =
          GREY_COLOR,
        linewidth = 0.65,
        lineend = "round"
      ) +
      geom_point(
        color =
          ORANGE_COLOR,
        shape = 18,
        size = 3.4,
        alpha = 0.95
      ) +
      scale_x_continuous(
        breaks =
          difference_breaks,
        labels = label_number(
          accuracy =
            PAIRED_DIFFERENCE_X_ACCURACY
        ),
        expand = expansion(
          mult = c(
            0.02,
            0.04
          )
        )
      ) +
      coord_cartesian(
        xlim =
          difference_limits,
        clip = "off"
      ) +
      labs(
        x =
          expression(
            Delta * rho ==
              rho[Confidence] -
              rho[w/o~Confidence]
          ),
        y = NULL,
        title =
          "Paired Improvement in Spearman Correlation",
        subtitle =
          subtitle_text,
        caption =
          "Points show release-pair differences. The orange vertical line and shaded band show the mean difference and its 95% confidence interval."
      ) +
      theme_minimal() +
      theme(
        text = element_text(
          family = "Arial"
        ),
        plot.background = element_rect(
          fill = "white",
          color = NA
        ),
        panel.background = element_rect(
          fill = "white",
          color = NA
        ),
        panel.grid.major.x =
          element_line(
            color = "#D7D7D7",
            linewidth = 0.50
          ),
        panel.grid.major.y =
          element_line(
            color = "#F1F1F1",
            linewidth = 0.35
          ),
        panel.grid.minor =
          element_blank(),
        plot.title =
          element_text(
            size = 18,
            face = "bold",
            color = "black",
            margin = margin(
              b = 6
            )
          ),
        plot.subtitle =
          element_text(
            size = 10.5,
            color = "#2F2F2F",
            lineheight = 1.20,
            margin = margin(
              b = 10
            )
          ),
        axis.title.x =
          element_text(
            size = 15,
            color = "black",
            margin = margin(
              t = 10
            )
          ),
        axis.text.x =
          element_text(
            size = 11.5,
            color = "black",
            margin = margin(
              t = 5
            )
          ),
        axis.text.y =
          element_text(
            size = 9.3,
            color = "black",
            margin = margin(
              r = 7
            )
          ),
        plot.caption.position =
          "plot",
        plot.caption =
          element_text(
            size = 9.5,
            color = "#333333",
            hjust = 0.5,
            margin = margin(
              t = 9
            )
          ),
        plot.margin = margin(
          t = 8,
          r = 18,
          b = 12,
          l = 18
        )
      )
  )
}


save_separate_figures <- function(
    distribution_plot,
    paired_difference_plot
)
{
  ggsave(
    filename =
      DISTRIBUTION_FIGURE_PATH,
    plot =
      distribution_plot,
    width =
      DISTRIBUTION_OUTPUT_WIDTH,
    height =
      DISTRIBUTION_OUTPUT_HEIGHT,
    units = "in",
    dpi =
      OUTPUT_DPI,
    device =
      grDevices::cairo_pdf,
    family = "Arial",
    limitsize = FALSE
  )
  
  ggsave(
    filename =
      PAIRED_DIFFERENCE_FIGURE_PATH,
    plot =
      paired_difference_plot,
    width =
      PAIRED_DIFFERENCE_OUTPUT_WIDTH,
    height =
      PAIRED_DIFFERENCE_OUTPUT_HEIGHT,
    units = "in",
    dpi =
      OUTPUT_DPI,
    device =
      grDevices::cairo_pdf,
    family = "Arial",
    limitsize = FALSE
  )
}

for (
  aggregation_method in
  CLEAR_AGGREGATION_METHODS
) {
  CLEAR_AGGREGATION_METHOD <- aggregation_method
  aggregation_label <- toupper(
    CLEAR_AGGREGATION_METHOD
  )
  
  METHOD_LABELS <- c(
    "without_confidence" =
      paste0(
        "CLEAR-",
        aggregation_label,
        " w/o Confidence"
      ),
    "with_confidence" =
      paste0(
        "CLEAR-",
        aggregation_label
      )
  )
  
  OUTPUT_ROOT <- file.path(
    ".",
    "result_fig",
    "RQ_ablation_confidence",
    DATASET_NAME,
    "adjacent_release_spearman_common_tokens",
    CLEAR_AGGREGATION_METHOD
  )
  
  DISTRIBUTION_FIGURE_PATH <- file.path(
    OUTPUT_ROOT,
    paste0(
      "RQ_confidence_",
      CLEAR_AGGREGATION_METHOD,
      "_adjacent_release_spearman_distribution.pdf"
    )
  )
  
  PAIRED_DIFFERENCE_FIGURE_PATH <- file.path(
    OUTPUT_ROOT,
    paste0(
      "RQ_confidence_",
      CLEAR_AGGREGATION_METHOD,
      "_adjacent_release_spearman_paired_difference.pdf"
    )
  )
  
  PAIRED_DIFFERENCE_RESULT_PATH <- file.path(
    OUTPUT_ROOT,
    paste0(
      "confidence_",
      CLEAR_AGGREGATION_METHOD,
      "_adjacent_release_spearman_paired_differences.csv"
    )
  )
  
  PAIRWISE_RESULT_PATH <- file.path(
    OUTPUT_ROOT,
    paste0(
      "confidence_",
      CLEAR_AGGREGATION_METHOD,
      "_adjacent_release_spearman_pairwise.csv"
    )
  )
  
  SUMMARY_RESULT_PATH <- file.path(
    OUTPUT_ROOT,
    paste0(
      "confidence_",
      CLEAR_AGGREGATION_METHOD,
      "_adjacent_release_spearman_summary.csv"
    )
  )
  
  PAIRED_COMPARISON_PATH <- file.path(
    OUTPUT_ROOT,
    paste0(
      "confidence_",
      CLEAR_AGGREGATION_METHOD,
      "_adjacent_release_spearman_paired_comparison.csv"
    )
  )
  
  ZERO_TEST_RESULT_PATH <- file.path(
    OUTPUT_ROOT,
    paste0(
      "confidence_",
      CLEAR_AGGREGATION_METHOD,
      "_adjacent_release_spearman_zero_tests.csv"
    )
  )
  
  dir.create(
    OUTPUT_ROOT,
    recursive = TRUE,
    showWarnings = FALSE
  )
  
  release_pairs <- build_release_pairs(
    LINEDP_PROJECTS
  )
  
  pairwise_results <- analyze_all_release_pairs(
    release_pairs
  )
  
  summary_results <- summarize_spearman(
    pairwise_results
  )
  
  paired_comparison <- calculate_paired_comparison(
    pairwise_results
  )
  
  zero_test_results <- calculate_zero_tests(
    pairwise_results
  )
  
  paired_difference_results <- prepare_paired_difference_data(
    pairwise_results
  )
  
  distribution_plot <- build_distribution_plot(
    result_data =
      pairwise_results,
    summary_data =
      summary_results,
    paired_comparison =
      paired_comparison,
    zero_test_data =
      zero_test_results
  )
  
  paired_difference_plot <- build_paired_difference_plot(
    paired_difference_data =
      paired_difference_results,
    paired_comparison =
      paired_comparison
  )
  
  write.csv(
    pairwise_results,
    PAIRWISE_RESULT_PATH,
    row.names = FALSE,
    fileEncoding = "UTF-8"
  )
  
  write.csv(
    summary_results,
    SUMMARY_RESULT_PATH,
    row.names = FALSE,
    fileEncoding = "UTF-8"
  )
  
  write.csv(
    paired_comparison,
    PAIRED_COMPARISON_PATH,
    row.names = FALSE,
    fileEncoding = "UTF-8"
  )
  
  write.csv(
    zero_test_results,
    ZERO_TEST_RESULT_PATH,
    row.names = FALSE,
    fileEncoding = "UTF-8"
  )
  
  write.csv(
    paired_difference_results,
    PAIRED_DIFFERENCE_RESULT_PATH,
    row.names = FALSE,
    fileEncoding = "UTF-8"
  )
  
  save_separate_figures(
    distribution_plot =
      distribution_plot,
    paired_difference_plot =
      paired_difference_plot
  )
  
  message(
    paste0(
      "[CLEAR-",
      aggregation_label,
      "] Saved the adjacent-release Spearman Distribution figure to ",
      DISTRIBUTION_FIGURE_PATH
    )
  )
  
  message(
    paste0(
      "[CLEAR-",
      aggregation_label,
      "] Saved the paired Spearman difference figure to ",
      PAIRED_DIFFERENCE_FIGURE_PATH
    )
  )
  
  message(
    paste0(
      "[CLEAR-",
      aggregation_label,
      "] Saved all adjacent-release Spearman statistics to ",
      OUTPUT_ROOT
    )
  )
}

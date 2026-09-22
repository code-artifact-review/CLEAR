suppressPackageStartupMessages({
  library(ggplot2)
  library(scales)
})

PROJECT_ROOT <- file.path("..", "..")

DATASET_NAME <- "linedp_dataset"
CLEAR_AGGREGATION_METHODS <- c(
  "sum",
  "max",
  "top3avg",
  "top2avg"
)

CLEAR_AGGREGATION_METHOD <- NULL
INSPECTION_RATIO <- 0.20
DEEPLINEDP_FOLD <- "0"
FAIL_ON_MISSING_FILE <- TRUE

CLEAR_ROOT <- file.path(
  PROJECT_ROOT,
  "clear",
  "Data",
  DATASET_NAME,
  "results",
  "ranking_evaluation",
  "logistic_regression"
)

SOUND_OP2_ROOT <- file.path(
  PROJECT_ROOT,
  "Result",
  DATASET_NAME,
  "Op2",
  "line_result"
)

GLANCE_LR_ROOT <- file.path(
  PROJECT_ROOT,
  "Result",
  DATASET_NAME,
  "BASE-Glance-LR_Mixed_Sort",
  "line_result"
)

LINEDP_ROOT <- file.path(
  PROJECT_ROOT,
  "Result",
  DATASET_NAME,
  "MIT-LineDP_mixedsort",
  "line_result"
)

NGRAM_ROOT <- file.path(
  PROJECT_ROOT,
  "baselines",
  "sound",
  "src",
  "models",
  "Ngram",
  "n_gram_result",
  DATASET_NAME
)

ERRORPRONE_ROOT <- file.path(
  PROJECT_ROOT,
  "baselines",
  "sound",
  "src",
  "models",
  "ErrorProne",
  "ErrorProne_result",
  DATASET_NAME
)

DEEPLINEDP_ROOT <- file.path(
  PROJECT_ROOT,
  "baselines",
  "sound",
  "output",
  "prediction",
  "DeepLineDP",
  "within-release",
  DATASET_NAME,
  DEEPLINEDP_FOLD
)

OUTPUT_ROOT <- file.path(
  PROJECT_ROOT,
  "clear",
  "exp",
  "result_fig",
  "Hit_Over",
  DATASET_NAME
)

PER_RELEASE_OUTPUT_PATH <- NULL
SUMMARY_OUTPUT_PATH <- NULL
FIGURE_OUTPUT_PATH <- NULL

TN_PER_RELEASE_OUTPUT_PATH <- NULL
TN_SUMMARY_OUTPUT_PATH <- NULL
TN_FIGURE_OUTPUT_PATH <- NULL
BASELINE_ORDER <- c(
  "SOUND-Op2",
  "GLANCE-LR",
  "LineDP",
  "Ngram",
  "DeepLineDP",
  "ErrorProne"
)

METRIC_ORDER <- c(
  "hit",
  "over"
)

METRIC_LABELS <- c(
  "hit" = "Hit",
  "over" = "Over"
)

METRIC_COLORS <- c(
  "hit" = "#547DA3",
  "over" = "#DA936A"
)

BAR_WIDTH <- 0.68
DODGE_WIDTH <- 0.78
OUTPUT_WIDTH <- 13
OUTPUT_HEIGHT <- 7
OUTPUT_DPI <- 300

LABEL_VERTICAL_RATIO <- 0.3

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


normalize_column_name <- function(
    column_name
)
{
  return(
    tolower(
      gsub(
        "[^A-Za-z0-9]+",
        "",
        column_name
      )
    )
  )
}


find_column <- function(
    dataframe,
    candidate_names,
    required = TRUE
)
{
  normalized_names <- vapply(
    names(
      dataframe
    ),
    normalize_column_name,
    character(1)
  )
  
  normalized_candidates <- vapply(
    candidate_names,
    normalize_column_name,
    character(1)
  )
  
  for (
    candidate_index in
    seq_along(
      normalized_candidates
    )
  ) {
    matched_index <- match(
      normalized_candidates[
        candidate_index
      ],
      normalized_names
    )
    
    if (!is.na(matched_index)) {
      return(
        names(
          dataframe
        )[
          matched_index
        ]
      )
    }
  }
  
  if (required) {
    stop(
      paste0(
        "None of the required columns were found: ",
        paste(
          candidate_names,
          collapse = ", "
        ),
        ". Available columns: ",
        paste(
          names(
            dataframe
          ),
          collapse = ", "
        ),
        "."
      )
    )
  }
  
  return(NULL)
}


detect_separator <- function(
    input_path
)
{
  header_line <- readLines(
    input_path,
    n = 1,
    warn = FALSE,
    encoding = "UTF-8"
  )
  
  if (length(header_line) == 0) {
    stop(
      paste0(
        "The input file is empty: ",
        input_path
      )
    )
  }
  
  tab_count <- lengths(
    gregexpr(
      "\t",
      header_line,
      fixed = TRUE
    )
  )
  
  comma_count <- lengths(
    gregexpr(
      ",",
      header_line,
      fixed = TRUE
    )
  )
  
  if (
    tab_count > comma_count &&
    tab_count > 0
  ) {
    return("\t")
  }
  
  return(",")
}


read_table_auto <- function(
    input_path
)
{
  if (!file.exists(input_path)) {
    stop(
      paste0(
        "Input file does not exist: ",
        input_path
      )
    )
  }
  
  separator <- detect_separator(
    input_path
  )
  
  read_attempt <- function(
    file_encoding
  )
  {
    return(
      read.table(
        input_path,
        header = TRUE,
        sep = separator,
        quote = "\"",
        comment.char = "",
        fill = TRUE,
        check.names = FALSE,
        stringsAsFactors = FALSE,
        fileEncoding = file_encoding
      )
    )
  }
  
  dataframe <- tryCatch(
    read_attempt(
      "UTF-8-BOM"
    ),
    error = function(
    first_error
    )
    {
      tryCatch(
        read_attempt(
          "UTF-8"
        ),
        error = function(
    second_error
        )
        {
          stop(
            paste0(
              "Failed to read ",
              input_path,
              ": ",
              conditionMessage(
                second_error
              )
            )
          )
        }
      )
    }
  )
  
  if (nrow(dataframe) == 0) {
    stop(
      paste0(
        "No data rows were found in ",
        input_path,
        "."
      )
    )
  }
  
  return(
    dataframe
  )
}


normalize_file_path <- function(
    file_path
)
{
  normalized_path <- gsub(
    "\\\\",
    "/",
    as.character(
      file_path
    )
  )
  
  normalized_path <- gsub(
    "/+",
    "/",
    normalized_path
  )
  
  normalized_path <- sub(
    "^\\./",
    "",
    normalized_path
  )
  
  normalized_path <- sub(
    "^/",
    "",
    normalized_path
  )
  
  return(
    normalized_path
  )
}


parse_binary_values <- function(
    values
)
{
  normalized_values <- tolower(
    trimws(
      as.character(
        values
      )
    )
  )
  
  output_values <- rep(
    NA_integer_,
    length(
      normalized_values
    )
  )
  
  output_values[
    normalized_values %in% c(
      "1",
      "true",
      "t",
      "yes",
      "y"
    )
  ] <- 1L
  
  output_values[
    normalized_values %in% c(
      "0",
      "false",
      "f",
      "no",
      "n"
    )
  ] <- 0L
  
  numeric_values <- suppressWarnings(
    as.numeric(
      normalized_values
    )
  )
  
  numeric_mask <- (
    is.na(
      output_values
    ) &
      is.finite(
        numeric_values
      )
  )
  
  output_values[
    numeric_mask
  ] <- as.integer(
    numeric_values[
      numeric_mask
    ] != 0
  )
  
  return(
    output_values
  )
}


build_line_key <- function(
    filename,
    line_number
)
{
  normalized_filename <- normalize_file_path(
    filename
  )
  
  numeric_line_number <- suppressWarnings(
    as.integer(
      as.character(
        line_number
      )
    )
  )
  
  return(
    paste0(
      normalized_filename,
      ":",
      numeric_line_number
    )
  )
}


parse_combined_line_identifier <- function(
    line_identifier
)
{
  identifier_values <- as.character(
    line_identifier
  )
  
  matched_parts <- regexec(
    "^(.*):([0-9]+)$",
    identifier_values
  )
  
  extracted_parts <- regmatches(
    identifier_values,
    matched_parts
  )
  
  valid_mask <- lengths(
    extracted_parts
  ) == 3
  
  filenames <- rep(
    NA_character_,
    length(
      identifier_values
    )
  )
  
  line_numbers <- rep(
    NA_integer_,
    length(
      identifier_values
    )
  )
  
  filenames[
    valid_mask
  ] <- vapply(
    extracted_parts[
      valid_mask
    ],
    function(
    parts
    )
    {
      return(
        parts[2]
      )
    },
    character(1)
  )
  
  line_numbers[
    valid_mask
  ] <- as.integer(
    vapply(
      extracted_parts[
        valid_mask
      ],
      function(
    parts
      )
      {
        return(
          parts[3]
        )
      },
    character(1)
    )
  )
  
  return(
    data.frame(
      filename =
        normalize_file_path(
          filenames
        ),
      line_number =
        line_numbers,
      stringsAsFactors = FALSE
    )
  )
}


collect_target_releases <- function(
    projects
)
{
  result_rows <- list()
  row_index <- 1
  
  for (project_name in names(projects)) {
    releases <- projects[[
      project_name
    ]]
    
    if (length(releases) < 2) {
      next
    }
    
    for (
      release_index in
      2:length(
        releases
      )
    ) {
      result_rows[[
        row_index
      ]] <- data.frame(
        project =
          project_name,
        train_release =
          releases[
            release_index -
              1
          ],
        test_release =
          releases[
            release_index
          ],
        stringsAsFactors = FALSE
      )
      
      row_index <- row_index + 1
    }
  }
  
  if (length(result_rows) == 0) {
    stop(
      "No target releases were configured."
    )
  }
  
  return(
    do.call(
      rbind,
      result_rows
    )
  )
}


get_clear_path <- function(
    test_release
)
{
  return(
    file.path(
      CLEAR_ROOT,
      CLEAR_AGGREGATION_METHOD,
      paste0(
        test_release,
        "_",
        CLEAR_AGGREGATION_METHOD,
        "_best_ranking.csv"
      )
    )
  )
}


get_sound_op2_path <- function(
    project,
    test_release
)
{
  return(
    file.path(
      SOUND_OP2_ROOT,
      project,
      paste0(
        test_release,
        "-result.csv"
      )
    )
  )
}


get_glance_lr_path <- function(
    project,
    test_release
)
{
  return(
    file.path(
      GLANCE_LR_ROOT,
      project,
      paste0(
        test_release,
        "-result.csv"
      )
    )
  )
}


get_linedp_path <- function(
    project,
    test_release
)
{
  return(
    file.path(
      LINEDP_ROOT,
      project,
      paste0(
        test_release,
        "-result.csv"
      )
    )
  )
}


get_ngram_path <- function(
    test_release
)
{
  return(
    file.path(
      NGRAM_ROOT,
      paste0(
        test_release,
        "-line-lvl-result.txt"
      )
    )
  )
}


get_errorprone_path <- function(
    test_release
)
{
  return(
    file.path(
      ERRORPRONE_ROOT,
      paste0(
        test_release,
        "-line-lvl-result.txt"
      )
    )
  )
}


get_deeplinedp_path <- function(
    test_release
)
{
  return(
    file.path(
      DEEPLINEDP_ROOT,
      paste0(
        test_release,
        ".csv"
      )
    )
  )
}


load_clear_reference <- function(
    input_path
)
{
  dataframe <- read_table_auto(
    input_path
  )
  
  filename_column <- find_column(
    dataframe,
    c(
      "filename",
      "file-name",
      "file_name"
    )
  )
  
  line_number_column <- find_column(
    dataframe,
    c(
      "original_line_number",
      "original-line-number",
      "original line number"
    )
  )
  
  label_column <- find_column(
    dataframe,
    c(
      "line-label",
      "line_level_ground_truth",
      "line-level-ground-truth"
    )
  )
  
  rank_column <- find_column(
    dataframe,
    c(
      "rank"
    )
  )
  
  output_data <- data.frame(
    filename =
      normalize_file_path(
        dataframe[[
          filename_column
        ]]
      ),
    line_number =
      suppressWarnings(
        as.integer(
          as.character(
            dataframe[[
              line_number_column
            ]]
          )
        )
      ),
    line_label =
      parse_binary_values(
        dataframe[[
          label_column
        ]]
      ),
    clear_rank =
      suppressWarnings(
        as.numeric(
          dataframe[[
            rank_column
          ]]
        )
      ),
    original_order =
      seq_len(
        nrow(
          dataframe
        )
      ),
    stringsAsFactors = FALSE
  )
  
  output_data$line_key <- build_line_key(
    output_data$filename,
    output_data$line_number
  )
  
  valid_mask <- (
    !is.na(
      output_data$filename
    ) &
      nzchar(
        output_data$filename
      ) &
      is.finite(
        output_data$line_number
      ) &
      !is.na(
        output_data$line_label
      ) &
      is.finite(
        output_data$clear_rank
      )
  )
  
  output_data <- output_data[
    valid_mask,
    ,
    drop = FALSE
  ]
  
  if (
    any(
      duplicated(
        output_data$line_key
      )
    )
  ) {
    duplicate_keys <- unique(
      output_data$line_key[
        duplicated(
          output_data$line_key
        )
      ]
    )
    
    stop(
      paste0(
        "Duplicate line identifiers were found in CLEAR ranking: ",
        paste(
          head(
            duplicate_keys,
            10
          ),
          collapse = ", "
        ),
        "."
      )
    )
  }
  
  output_data <- output_data[
    order(
      output_data$clear_rank,
      output_data$original_order,
      na.last = TRUE
    ),
    ,
    drop = FALSE
  ]
  
  output_data$canonical_order <- seq_len(
    nrow(
      output_data
    )
  )
  
  return(
    output_data
  )
}


load_generic_line_result <- function(
    input_path
)
{
  dataframe <- read_table_auto(
    input_path
  )
  
  line_identifier_column <- find_column(
    dataframe,
    c(
      "predicted_buggy_lines",
      "predicted-buggy-lines",
      "line_id",
      "line-id"
    )
  )
  
  score_column <- find_column(
    dataframe,
    c(
      "predicted_buggy_score",
      "predicted-buggy-score",
      "line_score",
      "line-score",
      "score"
    )
  )
  
  parsed_identifiers <- (
    parse_combined_line_identifier(
      dataframe[[
        line_identifier_column
      ]]
    )
  )
  
  output_data <- data.frame(
    filename =
      parsed_identifiers$filename,
    line_number =
      parsed_identifiers$line_number,
    line_score =
      suppressWarnings(
        as.numeric(
          dataframe[[
            score_column
          ]]
        )
      ),
    source_order =
      seq_len(
        nrow(
          dataframe
        )
      ),
    stringsAsFactors = FALSE
  )
  
  output_data$line_key <- build_line_key(
    output_data$filename,
    output_data$line_number
  )
  
  valid_mask <- (
    !is.na(
      output_data$filename
    ) &
      nzchar(
        output_data$filename
      ) &
      is.finite(
        output_data$line_number
      ) &
      is.finite(
        output_data$line_score
      )
  )
  
  output_data <- output_data[
    valid_mask,
    ,
    drop = FALSE
  ]
  
  output_data <- output_data[
    order(
      -output_data$line_score,
      output_data$source_order,
      na.last = TRUE
    ),
    ,
    drop = FALSE
  ]
  
  output_data <- output_data[
    !duplicated(
      output_data$line_key
    ),
    ,
    drop = FALSE
  ]
  
  return(
    output_data
  )
}


load_deeplinedp_data <- function(
    input_path
)
{
  dataframe <- read_table_auto(
    input_path
  )
  
  filename_column <- find_column(
    dataframe,
    c(
      "filename",
      "file-name",
      "file_name"
    )
  )
  
  line_number_column <- find_column(
    dataframe,
    c(
      "line-number",
      "line_number"
    )
  )
  
  file_probability_column <- find_column(
    dataframe,
    c(
      "prediction-prob",
      "prediction_prob",
      "file_prediction_probability"
    )
  )
  
  file_label_column <- find_column(
    dataframe,
    c(
      "prediction-label",
      "prediction_label",
      "file_prediction_label"
    )
  )
  
  line_score_column <- find_column(
    dataframe,
    c(
      "line-attention-score",
      "line_attention_score",
      "line-score",
      "line_score"
    )
  )
  
  output_data <- data.frame(
    filename =
      normalize_file_path(
        dataframe[[
          filename_column
        ]]
      ),
    line_number =
      suppressWarnings(
        as.integer(
          as.character(
            dataframe[[
              line_number_column
            ]]
          )
        )
      ),
    file_prediction_probability =
      suppressWarnings(
        as.numeric(
          dataframe[[
            file_probability_column
          ]]
        )
      ),
    file_prediction_label =
      parse_binary_values(
        dataframe[[
          file_label_column
        ]]
      ),
    line_score =
      suppressWarnings(
        as.numeric(
          dataframe[[
            line_score_column
          ]]
        )
      ),
    source_order =
      seq_len(
        nrow(
          dataframe
        )
      ),
    stringsAsFactors = FALSE
  )
  
  output_data$line_key <- build_line_key(
    output_data$filename,
    output_data$line_number
  )
  
  valid_mask <- (
    !is.na(
      output_data$filename
    ) &
      nzchar(
        output_data$filename
      ) &
      is.finite(
        output_data$line_number
      )
  )
  
  output_data <- output_data[
    valid_mask,
    ,
    drop = FALSE
  ]
  
  output_data$file_prediction_probability[
    !is.finite(
      output_data$file_prediction_probability
    )
  ] <- -Inf
  
  output_data$file_prediction_label[
    is.na(
      output_data$file_prediction_label
    )
  ] <- 0L
  
  output_data$line_score[
    !is.finite(
      output_data$line_score
    )
  ] <- -Inf
  
  output_data <- output_data[
    order(
      -output_data$file_prediction_label,
      -output_data$file_prediction_probability,
      -output_data$line_score,
      output_data$source_order,
      na.last = TRUE
    ),
    ,
    drop = FALSE
  ]
  
  line_data <- output_data[
    !duplicated(
      output_data$line_key
    ),
    ,
    drop = FALSE
  ]
  
  file_data <- output_data[
    order(
      output_data$filename,
      -output_data$file_prediction_label,
      -output_data$file_prediction_probability,
      output_data$source_order,
      na.last = TRUE
    ),
    ,
    drop = FALSE
  ]
  
  file_data <- file_data[
    !duplicated(
      file_data$filename
    ),
    c(
      "filename",
      "file_prediction_label",
      "file_prediction_probability"
    ),
    drop = FALSE
  ]
  
  return(
    list(
      line_data =
        line_data,
      file_data =
        file_data
    )
  )
}


load_ngram_lines <- function(
    input_path
)
{
  dataframe <- read_table_auto(
    input_path
  )
  
  filename_column <- find_column(
    dataframe,
    c(
      "file-name",
      "filename",
      "file_name"
    )
  )
  
  line_number_column <- find_column(
    dataframe,
    c(
      "line-number",
      "line_number"
    )
  )
  
  line_score_column <- find_column(
    dataframe,
    c(
      "line-score",
      "line_score"
    )
  )
  
  output_data <- data.frame(
    filename =
      normalize_file_path(
        dataframe[[
          filename_column
        ]]
      ),
    line_number =
      suppressWarnings(
        as.integer(
          as.character(
            dataframe[[
              line_number_column
            ]]
          )
        )
      ),
    line_score =
      suppressWarnings(
        as.numeric(
          dataframe[[
            line_score_column
          ]]
        )
      ),
    source_order =
      seq_len(
        nrow(
          dataframe
        )
      ),
    stringsAsFactors = FALSE
  )
  
  output_data$line_key <- build_line_key(
    output_data$filename,
    output_data$line_number
  )
  
  valid_mask <- (
    !is.na(
      output_data$filename
    ) &
      nzchar(
        output_data$filename
      ) &
      is.finite(
        output_data$line_number
      ) &
      is.finite(
        output_data$line_score
      )
  )
  
  output_data <- output_data[
    valid_mask,
    ,
    drop = FALSE
  ]
  
  output_data <- output_data[
    order(
      -output_data$line_score,
      output_data$source_order,
      na.last = TRUE
    ),
    ,
    drop = FALSE
  ]
  
  output_data <- output_data[
    !duplicated(
      output_data$line_key
    ),
    ,
    drop = FALSE
  ]
  
  return(
    output_data
  )
}


load_errorprone_lines <- function(
    input_path
)
{
  dataframe <- read_table_auto(
    input_path
  )
  
  filename_column <- find_column(
    dataframe,
    c(
      "filename",
      "file-name",
      "file_name"
    )
  )
  
  line_number_column <- find_column(
    dataframe,
    c(
      "line_number",
      "line-number"
    )
  )
  
  prediction_column <- find_column(
    dataframe,
    c(
      "EP_prediction_result",
      "ep-prediction-result",
      "prediction"
    )
  )
  
  output_data <- data.frame(
    filename =
      normalize_file_path(
        dataframe[[
          filename_column
        ]]
      ),
    line_number =
      suppressWarnings(
        as.integer(
          as.character(
            dataframe[[
              line_number_column
            ]]
          )
        )
      ),
    line_score =
      parse_binary_values(
        dataframe[[
          prediction_column
        ]]
      ),
    source_order =
      seq_len(
        nrow(
          dataframe
        )
      ),
    stringsAsFactors = FALSE
  )
  
  output_data$line_key <- build_line_key(
    output_data$filename,
    output_data$line_number
  )
  
  valid_mask <- (
    !is.na(
      output_data$filename
    ) &
      nzchar(
        output_data$filename
      ) &
      is.finite(
        output_data$line_number
      ) &
      !is.na(
        output_data$line_score
      )
  )
  
  output_data <- output_data[
    valid_mask,
    ,
    drop = FALSE
  ]
  
  output_data <- output_data[
    order(
      -output_data$line_score,
      output_data$source_order,
      na.last = TRUE
    ),
    ,
    drop = FALSE
  ]
  
  output_data <- output_data[
    !duplicated(
      output_data$line_key
    ),
    ,
    drop = FALSE
  ]
  
  return(
    output_data
  )
}


rank_generic_method <- function(
    canonical_data,
    method_line_data
)
{
  matched_index <- match(
    canonical_data$line_key,
    method_line_data$line_key
  )
  
  ranking_data <- canonical_data
  ranking_data$method_score <- (
    method_line_data$line_score[
      matched_index
    ]
  )
  
  matched_mask <- is.finite(
    ranking_data$method_score
  )
  
  ranking_data$method_score[
    !matched_mask
  ] <- -Inf
  
  ranking_order <- order(
    -ranking_data$method_score,
    ranking_data$canonical_order,
    na.last = TRUE
  )
  
  return(
    ranking_data[
      ranking_order,
      ,
      drop = FALSE
    ]
  )
}

rank_file_first_method <- function(
    canonical_data,
    method_line_data,
    deep_file_data
)
{
  line_matched_index <- match(
    canonical_data$line_key,
    method_line_data$line_key
  )
  
  file_matched_index <- match(
    canonical_data$filename,
    deep_file_data$filename
  )
  
  ranking_data <- canonical_data
  
  ranking_data$method_score <- (
    method_line_data$line_score[
      line_matched_index
    ]
  )
  
  ranking_data$file_prediction_label <- (
    deep_file_data$file_prediction_label[
      file_matched_index
    ]
  )
  
  ranking_data$file_prediction_probability <- (
    deep_file_data$file_prediction_probability[
      file_matched_index
    ]
  )
  
  matched_line_mask <- is.finite(
    ranking_data$method_score
  )
  
  ranking_data$method_score[
    !matched_line_mask
  ] <- -Inf
  
  ranking_data$file_prediction_label[
    is.na(
      ranking_data$file_prediction_label
    )
  ] <- 0L
  
  ranking_data$file_prediction_probability[
    !is.finite(
      ranking_data$file_prediction_probability
    )
  ] <- -Inf
  
  ranking_order <- order(
    -ranking_data$file_prediction_label,
    -ranking_data$file_prediction_probability,
    -ranking_data$method_score,
    ranking_data$canonical_order,
    na.last = TRUE
  )
  
  return(
    ranking_data[
      ranking_order,
      ,
      drop = FALSE
    ]
  )
}

get_top_true_positive_set <- function(
    ranked_data
)
{
  total_lines <- nrow(
    ranked_data
  )
  
  if (total_lines == 0) {
    stop(
      "Cannot inspect an empty ranking."
    )
  }
  
  inspection_count <- max(
    1,
    floor(
      total_lines *
        INSPECTION_RATIO
    )
  )
  
  inspection_count <- min(
    inspection_count,
    total_lines
  )
  
  inspected_data <- ranked_data[
    seq_len(
      inspection_count
    ),
    ,
    drop = FALSE
  ]
  
  true_positive_set <- unique(
    inspected_data$line_key[
      inspected_data$line_label == 1
    ]
  )
  
  return(
    list(
      true_positive_set =
        true_positive_set,
      inspection_count =
        inspection_count,
      total_lines =
        total_lines
    )
  )
}


get_true_negative_set <- function(
    ranked_data
)
{
  total_lines <- nrow(
    ranked_data
  )
  
  if (total_lines == 0) {
    stop(
      "Cannot inspect an empty ranking."
    )
  }
  
  inspection_count <- max(
    1,
    floor(
      total_lines *
        INSPECTION_RATIO
    )
  )
  
  inspection_count <- min(
    inspection_count,
    total_lines
  )
  
  if (inspection_count >= total_lines) {
    true_negative_set <- character(0)
  } else {
    uninspected_data <- ranked_data[
      (inspection_count + 1):total_lines,
      ,
      drop = FALSE
    ]
    
    true_negative_set <- unique(
      uninspected_data$line_key[
        uninspected_data$line_label == 0
      ]
    )
  }
  
  return(
    list(
      true_negative_set =
        true_negative_set,
      inspection_count =
        inspection_count,
      total_lines =
        total_lines
    )
  )
}


calculate_hit_over <- function(
    clear_set,
    baseline_set
)
{
  common_set <- intersect(
    clear_set,
    baseline_set
  )
  
  clear_only_set <- setdiff(
    clear_set,
    baseline_set
  )
  
  baseline_only_set <- setdiff(
    baseline_set,
    clear_set
  )
  
  baseline_count <- length(
    baseline_set
  )
  
  hit <- if (
    baseline_count == 0
  ) {
    NA_real_
  } else {
    length(
      common_set
    ) /
      baseline_count
  }
  
  over <- if (
    baseline_count == 0
  ) {
    NA_real_
  } else {
    length(
      clear_only_set
    ) /
      baseline_count
  }
  
  return(
    list(
      clear_count =
        length(
          clear_set
        ),
      baseline_count =
        baseline_count,
      common_count =
        length(
          common_set
        ),
      clear_only_count =
        length(
          clear_only_set
        ),
      baseline_only_count =
        length(
          baseline_only_set
        ),
      hit =
        hit,
      over =
        over
    )
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
  
  critical_value <- qt(
    1 -
      (
        1 -
          confidence_level
      ) /
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


handle_missing_input <- function(
    input_path,
    method_name,
    test_release
)
{
  error_message <- paste0(
    "[",
    test_release,
    "][",
    method_name,
    "] Input file does not exist: ",
    input_path
  )
  
  if (FAIL_ON_MISSING_FILE) {
    stop(
      error_message
    )
  }
  
  warning(
    error_message
  )
  
  return(FALSE)
}


evaluate_one_release <- function(
    project,
    train_release,
    test_release
)
{
  message(
    paste0(
      "[",
      test_release,
      "] Loading CLEAR-",
      toupper(CLEAR_AGGREGATION_METHOD),
      " reference ranking."
    )
  )
  
  clear_path <- get_clear_path(
    test_release
  )
  
  if (!file.exists(clear_path)) {
    handle_missing_input(
      clear_path,
      paste0(
        "CLEAR-",
        toupper(CLEAR_AGGREGATION_METHOD)
      ),
      test_release
    )
    
    return(NULL)
  }
  
  canonical_data <- load_clear_reference(
    clear_path
  )
  
  clear_top_result <- get_top_true_positive_set(
    canonical_data
  )
  
  clear_tn_result <- get_true_negative_set(
    canonical_data
  )
  
  deep_path <- get_deeplinedp_path(
    test_release
  )
  
  if (!file.exists(deep_path)) {
    handle_missing_input(
      deep_path,
      "DeepLineDP",
      test_release
    )
    
    return(NULL)
  }
  
  deep_data <- load_deeplinedp_data(
    deep_path
  )
  
  method_results <- list()
  
  method_results[[
    "SOUND-Op2"
  ]] <- rank_generic_method(
    canonical_data,
    load_generic_line_result(
      get_sound_op2_path(
        project,
        test_release
      )
    )
  )
  
  method_results[[
    "GLANCE-LR"
  ]] <- rank_generic_method(
    canonical_data,
    load_generic_line_result(
      get_glance_lr_path(
        project,
        test_release
      )
    )
  )
  
  method_results[[
    "LineDP"
  ]] <- rank_generic_method(
    canonical_data,
    load_generic_line_result(
      get_linedp_path(
        project,
        test_release
      )
    )
  )
  
  method_results[[
    "Ngram"
  ]] <- rank_file_first_method(
    canonical_data,
    load_ngram_lines(
      get_ngram_path(
        test_release
      )
    ),
    deep_data$file_data
  )
  
  method_results[[
    "DeepLineDP"
  ]] <- rank_file_first_method(
    canonical_data,
    deep_data$line_data,
    deep_data$file_data
  )
  
  method_results[[
    "ErrorProne"
  ]] <- rank_file_first_method(
    canonical_data,
    load_errorprone_lines(
      get_errorprone_path(
        test_release
      )
    ),
    deep_data$file_data
  )
  
  result_rows <- list()
  tn_result_rows <- list()
  
  for (
    method_index in
    seq_along(
      BASELINE_ORDER
    )
  ) {
    method_name <- BASELINE_ORDER[
      method_index
    ]
    
    method_ranking <- method_results[[
      method_name
    ]]
    
    baseline_top_result <- get_top_true_positive_set(
      method_ranking
    )
    
    baseline_tn_result <- get_true_negative_set(
      method_ranking
    )
    
    comparison <- calculate_hit_over(
      clear_top_result$true_positive_set,
      baseline_top_result$true_positive_set
    )
    
    tn_comparison <- calculate_hit_over(
      clear_tn_result$true_negative_set,
      baseline_tn_result$true_negative_set
    )
    
    result_rows[[
      method_index
    ]] <- data.frame(
      project =
        project,
      train_release =
        train_release,
      test_release =
        test_release,
      baseline =
        method_name,
      inspection_ratio =
        INSPECTION_RATIO,
      total_lines =
        clear_top_result$total_lines,
      inspection_count =
        clear_top_result$inspection_count,
      clear_tp =
        comparison$clear_count,
      baseline_tp =
        comparison$baseline_count,
      common_tp =
        comparison$common_count,
      clear_only_tp =
        comparison$clear_only_count,
      baseline_only_tp =
        comparison$baseline_only_count,
      hit =
        comparison$hit,
      over =
        comparison$over,
      stringsAsFactors = FALSE
    )
    
    tn_result_rows[[
      method_index
    ]] <- data.frame(
      project =
        project,
      train_release =
        train_release,
      test_release =
        test_release,
      baseline =
        method_name,
      inspection_ratio =
        INSPECTION_RATIO,
      total_lines =
        clear_tn_result$total_lines,
      inspection_count =
        clear_tn_result$inspection_count,
      clear_tn =
        tn_comparison$clear_count,
      baseline_tn =
        tn_comparison$baseline_count,
      common_tn =
        tn_comparison$common_count,
      clear_only_tn =
        tn_comparison$clear_only_count,
      baseline_only_tn =
        tn_comparison$baseline_only_count,
      hit =
        tn_comparison$hit,
      over =
        tn_comparison$over,
      stringsAsFactors = FALSE
    )
  }
  
  return(
    list(
      results =
        do.call(
          rbind,
          result_rows
        ),
      tn_results =
        do.call(
          rbind,
          tn_result_rows
        )
    )
  )
}

summarize_results <- function(
    per_release_data
)
{
  summary_rows <- list()
  row_index <- 1
  
  for (baseline_name in BASELINE_ORDER) {
    baseline_data <- per_release_data[
      per_release_data$baseline ==
        baseline_name,
      ,
      drop = FALSE
    ]
    
    micro_denominator <- sum(
      baseline_data$baseline_tp
    )
    
    micro_hit <- if (
      micro_denominator == 0
    ) {
      NA_real_
    } else {
      sum(
        baseline_data$common_tp
      ) /
        micro_denominator
    }
    
    micro_over <- if (
      micro_denominator == 0
    ) {
      NA_real_
    } else {
      sum(
        baseline_data$clear_only_tp
      ) /
        micro_denominator
    }
    
    for (metric_name in METRIC_ORDER) {
      metric_values <- suppressWarnings(
        as.numeric(
          baseline_data[[
            metric_name
          ]]
        )
      )
      
      metric_values <- metric_values[
        is.finite(
          metric_values
        )
      ]
      
      confidence_interval <- (
        calculate_t_confidence_interval(
          metric_values
        )
      )
      
      summary_rows[[
        row_index
      ]] <- data.frame(
        baseline =
          baseline_name,
        metric =
          metric_name,
        total_release_count =
          nrow(
            baseline_data
          ),
        valid_release_count =
          length(
            metric_values
          ),
        mean =
          if (
            length(
              metric_values
            ) == 0
          ) {
            NA_real_
          } else {
            mean(
              metric_values
            )
          },
        median =
          if (
            length(
              metric_values
            ) == 0
          ) {
            NA_real_
          } else {
            median(
              metric_values
            )
          },
        standard_deviation =
          if (
            length(
              metric_values
            ) <= 1
          ) {
            0
          } else {
            sd(
              metric_values
            )
          },
        ci_lower =
          confidence_interval[1],
        ci_upper =
          confidence_interval[2],
        micro_value =
          if (
            metric_name == "hit"
          ) {
            micro_hit
          } else {
            micro_over
          },
        stringsAsFactors = FALSE
      )
      
      row_index <- row_index + 1
    }
  }
  
  summary_data <- do.call(
    rbind,
    summary_rows
  )
  
  summary_data$baseline <- factor(
    summary_data$baseline,
    levels = BASELINE_ORDER
  )
  
  summary_data$metric <- factor(
    summary_data$metric,
    levels = METRIC_ORDER
  )
  
  summary_data <- summary_data[
    order(
      summary_data$baseline,
      summary_data$metric
    ),
    ,
    drop = FALSE
  ]
  
  return(
    summary_data
  )
}


build_summary_plot <- function(
    summary_data
)
{
  plot_data <- summary_data
  plot_data$baseline <- factor(
    plot_data$baseline,
    levels = BASELINE_ORDER
  )
  plot_data$metric <- factor(
    plot_data$metric,
    levels = METRIC_ORDER
  )
  
  return(
    ggplot(
      plot_data,
      aes(
        x =
          baseline,
        y =
          mean,
        fill =
          metric,
        group =
          metric
      )
    ) +
      geom_col(
        position = position_dodge(
          width =
            DODGE_WIDTH
        ),
        width =
          BAR_WIDTH,
        color = "#3F3F3F",
        linewidth = 0.35,
        na.rm = TRUE
      ) +
      geom_errorbar(
        aes(
          ymin =
            ci_lower,
          ymax =
            ci_upper
        ),
        position = position_dodge(
          width =
            DODGE_WIDTH
        ),
        width = 0.13,
        linewidth = 0.75,
        color = "black",
        na.rm = TRUE
      ) +
      geom_text(
        aes(
          y =
            mean *
            LABEL_VERTICAL_RATIO,
          label = label_number(
            accuracy = 0.001
          )(
            mean
          )
        ),
        position = position_dodge(
          width =
            DODGE_WIDTH
        ),
        vjust = 0.5,
        size = 4.1,
        color = "black",
        na.rm = TRUE
      ) +
      scale_fill_manual(
        name = NULL,
        values =
          METRIC_COLORS,
        breaks =
          METRIC_ORDER,
        labels =
          METRIC_LABELS[
            METRIC_ORDER
          ],
        drop = FALSE
      ) +
      scale_x_discrete(
        limits =
          BASELINE_ORDER,
        drop = FALSE
      ) +
      scale_y_continuous(
        limits = c(
          0,
          NA
        ),
        breaks = pretty_breaks(
          n = 6
        ),
        labels = label_number(
          accuracy = 0.1
        ),
        expand = expansion(
          mult = c(
            0,
            0.10
          )
        )
      ) +
      labs(
        x = NULL,
        y = "Ratio relative to baseline true positives",
        title =
          paste0(
            "Hit and Over Defective Lines Identified by CLEAR-",
            toupper(CLEAR_AGGREGATION_METHOD)
          ),
        subtitle =
          "Top 20% of ranked lines; error bars show 95% t-based confidence intervals across releases."
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
        panel.grid.minor =
          element_blank(),
        panel.grid.major.y =
          element_line(
            color = "#D7D7D7",
            linewidth = 0.50
          ),
        plot.title =
          element_text(
            size = 20,
            face = "bold",
            color = "black",
            hjust = 0.5,
            margin = margin(
              b = 7
            )
          ),
        plot.subtitle =
          element_text(
            size = 11.5,
            color = "#2F2F2F",
            hjust = 0.5,
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
            size = 13.5,
            color = "black",
            angle = 20,
            hjust = 1,
            margin = margin(
              t = 8
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
        legend.position =
          "bottom",
        legend.text =
          element_text(
            size = 14,
            color = "black"
          ),
        legend.key.width =
          grid::unit(
            1.1,
            "cm"
          ),
        legend.key.height =
          grid::unit(
            0.55,
            "cm"
          ),
        legend.spacing.x =
          grid::unit(
            0.35,
            "cm"
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


summarize_tn_results <- function(
    per_release_data
)
{
  summary_rows <- list()
  row_index <- 1
  
  for (baseline_name in BASELINE_ORDER) {
    baseline_data <- per_release_data[
      per_release_data$baseline ==
        baseline_name,
      ,
      drop = FALSE
    ]
    
    micro_denominator <- sum(
      baseline_data$baseline_tn
    )
    
    micro_hit <- if (
      micro_denominator == 0
    ) {
      NA_real_
    } else {
      sum(
        baseline_data$common_tn
      ) /
        micro_denominator
    }
    
    micro_over <- if (
      micro_denominator == 0
    ) {
      NA_real_
    } else {
      sum(
        baseline_data$clear_only_tn
      ) /
        micro_denominator
    }
    
    for (metric_name in METRIC_ORDER) {
      metric_values <- suppressWarnings(
        as.numeric(
          baseline_data[[
            metric_name
          ]]
        )
      )
      
      metric_values <- metric_values[
        is.finite(
          metric_values
        )
      ]
      
      confidence_interval <- (
        calculate_t_confidence_interval(
          metric_values
        )
      )
      
      summary_rows[[
        row_index
      ]] <- data.frame(
        baseline =
          baseline_name,
        metric =
          metric_name,
        total_release_count =
          nrow(
            baseline_data
          ),
        valid_release_count =
          length(
            metric_values
          ),
        mean =
          if (
            length(
              metric_values
            ) == 0
          ) {
            NA_real_
          } else {
            mean(
              metric_values
            )
          },
        median =
          if (
            length(
              metric_values
            ) == 0
          ) {
            NA_real_
          } else {
            median(
              metric_values
            )
          },
        standard_deviation =
          if (
            length(
              metric_values
            ) <= 1
          ) {
            0
          } else {
            sd(
              metric_values
            )
          },
        ci_lower =
          confidence_interval[1],
        ci_upper =
          confidence_interval[2],
        micro_value =
          if (
            metric_name == "hit"
          ) {
            micro_hit
          } else {
            micro_over
          },
        stringsAsFactors = FALSE
      )
      
      row_index <- row_index + 1
    }
  }
  
  summary_data <- do.call(
    rbind,
    summary_rows
  )
  
  summary_data$baseline <- factor(
    summary_data$baseline,
    levels = BASELINE_ORDER
  )
  
  summary_data$metric <- factor(
    summary_data$metric,
    levels = METRIC_ORDER
  )
  
  summary_data <- summary_data[
    order(
      summary_data$baseline,
      summary_data$metric
    ),
    ,
    drop = FALSE
  ]
  
  return(
    summary_data
  )
}


build_tn_summary_plot <- function(
    summary_data
)
{
  plot_data <- summary_data
  plot_data$baseline <- factor(
    plot_data$baseline,
    levels = BASELINE_ORDER
  )
  plot_data$metric <- factor(
    plot_data$metric,
    levels = METRIC_ORDER
  )
  
  return(
    ggplot(
      plot_data,
      aes(
        x =
          baseline,
        y =
          mean,
        fill =
          metric,
        group =
          metric
      )
    ) +
      geom_col(
        position = position_dodge(
          width =
            DODGE_WIDTH
        ),
        width =
          BAR_WIDTH,
        color = "#3F3F3F",
        linewidth = 0.35,
        na.rm = TRUE
      ) +
      geom_errorbar(
        aes(
          ymin =
            ci_lower,
          ymax =
            ci_upper
        ),
        position = position_dodge(
          width =
            DODGE_WIDTH
        ),
        width = 0.13,
        linewidth = 0.75,
        color = "black",
        na.rm = TRUE
      ) +
      geom_text(
        aes(
          y =
            mean *
            LABEL_VERTICAL_RATIO,
          label = label_number(
            accuracy = 0.001
          )(
            mean
          )
        ),
        position = position_dodge(
          width =
            DODGE_WIDTH
        ),
        vjust = 0.5,
        size = 4.1,
        color = "black",
        na.rm = TRUE
      ) +
      scale_fill_manual(
        name = NULL,
        values =
          METRIC_COLORS,
        breaks =
          METRIC_ORDER,
        labels =
          METRIC_LABELS[
            METRIC_ORDER
          ],
        drop = FALSE
      ) +
      scale_x_discrete(
        limits =
          BASELINE_ORDER,
        drop = FALSE
      ) +
      scale_y_continuous(
        limits = c(
          0,
          NA
        ),
        breaks = pretty_breaks(
          n = 6
        ),
        labels = label_number(
          accuracy = 0.1
        ),
        expand = expansion(
          mult = c(
            0,
            0.10
          )
        )
      ) +
      labs(
        x = NULL,
        y = "Ratio relative to baseline true negatives",
        title =
          paste0(
            "Hit and Over True-Negative Lines Identified by CLEAR-",
            toupper(CLEAR_AGGREGATION_METHOD)
          ),
        subtitle =
          "Bottom 80% of ranked lines; error bars show 95% t-based confidence intervals across releases."
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
        panel.grid.minor =
          element_blank(),
        panel.grid.major.y =
          element_line(
            color = "#D7D7D7",
            linewidth = 0.50
          ),
        plot.title =
          element_text(
            size = 20,
            face = "bold",
            color = "black",
            hjust = 0.5,
            margin = margin(
              b = 7
            )
          ),
        plot.subtitle =
          element_text(
            size = 11.5,
            color = "#2F2F2F",
            hjust = 0.5,
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
            size = 13.5,
            color = "black",
            angle = 20,
            hjust = 1,
            margin = margin(
              t = 8
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
        legend.position =
          "bottom",
        legend.text =
          element_text(
            size = 14,
            color = "black"
          ),
        legend.key.width =
          grid::unit(
            1.1,
            "cm"
          ),
        legend.key.height =
          grid::unit(
            0.55,
            "cm"
          ),
        legend.spacing.x =
          grid::unit(
            0.35,
            "cm"
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


dir.create(
  OUTPUT_ROOT,
  recursive = TRUE,
  showWarnings = FALSE
)

for (
  aggregation_method in
  CLEAR_AGGREGATION_METHODS
) {
  CLEAR_AGGREGATION_METHOD <- aggregation_method
  aggregation_label <- toupper(
    CLEAR_AGGREGATION_METHOD
  )
  
  PER_RELEASE_OUTPUT_PATH <- file.path(
    OUTPUT_ROOT,
    paste0(
      "CLEAR_",
      aggregation_label,
      "_baseline_hit_over_per_release.csv"
    )
  )
  
  SUMMARY_OUTPUT_PATH <- file.path(
    OUTPUT_ROOT,
    paste0(
      "CLEAR_",
      aggregation_label,
      "_baseline_hit_over_summary.csv"
    )
  )
  
  FIGURE_OUTPUT_PATH <- file.path(
    OUTPUT_ROOT,
    paste0(
      "CLEAR_",
      aggregation_label,
      "_baseline_hit_over.pdf"
    )
  )
  
  TN_PER_RELEASE_OUTPUT_PATH <- file.path(
    OUTPUT_ROOT,
    paste0(
      "CLEAR_",
      aggregation_label,
      "_baseline_tn_hit_over_per_release.csv"
    )
  )
  
  TN_SUMMARY_OUTPUT_PATH <- file.path(
    OUTPUT_ROOT,
    paste0(
      "CLEAR_",
      aggregation_label,
      "_baseline_tn_hit_over_summary.csv"
    )
  )
  
  TN_FIGURE_OUTPUT_PATH <- file.path(
    OUTPUT_ROOT,
    paste0(
      "CLEAR_",
      aggregation_label,
      "_baseline_tn_hit_over.pdf"
    )
  )
  
  release_table <- collect_target_releases(
    LINEDP_PROJECTS
  )
  
  all_result_rows <- list()
  all_tn_result_rows <- list()
  successful_release_count <- 0
  
  for (
    release_index in
    seq_len(
      nrow(
        release_table
      )
    )
  ) {
    release_row <- release_table[
      release_index,
      ,
      drop = FALSE
    ]
    
    release_result <- evaluate_one_release(
      project =
        release_row$project,
      train_release =
        release_row$train_release,
      test_release =
        release_row$test_release
    )
    
    if (is.null(release_result)) {
      next
    }
    
    successful_release_count <- (
      successful_release_count +
        1
    )
    
    all_result_rows[[
      successful_release_count
    ]] <- release_result$results
    
    all_tn_result_rows[[
      successful_release_count
    ]] <- release_result$tn_results
  }
  
  if (successful_release_count == 0) {
    stop(
      "No release was successfully evaluated."
    )
  }
  
  per_release_results <- do.call(
    rbind,
    all_result_rows
  )
  
  tn_per_release_results <- do.call(
    rbind,
    all_tn_result_rows
  )
  
  summary_results <- summarize_results(
    per_release_results
  )
  
  tn_summary_results <- summarize_tn_results(
    tn_per_release_results
  )
  
  write.csv(
    per_release_results,
    PER_RELEASE_OUTPUT_PATH,
    row.names = FALSE,
    fileEncoding = "UTF-8"
  )
  
  write.csv(
    summary_results,
    SUMMARY_OUTPUT_PATH,
    row.names = FALSE,
    fileEncoding = "UTF-8"
  )
  
  write.csv(
    tn_per_release_results,
    TN_PER_RELEASE_OUTPUT_PATH,
    row.names = FALSE,
    fileEncoding = "UTF-8"
  )
  
  write.csv(
    tn_summary_results,
    TN_SUMMARY_OUTPUT_PATH,
    row.names = FALSE,
    fileEncoding = "UTF-8"
  )
  
  summary_plot <- build_summary_plot(
    summary_results
  )
  
  ggsave(
    filename =
      FIGURE_OUTPUT_PATH,
    plot =
      summary_plot,
    width =
      OUTPUT_WIDTH,
    height =
      OUTPUT_HEIGHT,
    units = "in",
    dpi =
      OUTPUT_DPI,
    device =
      grDevices::cairo_pdf,
    family = "Arial",
    limitsize = FALSE
  )
  
  tn_summary_plot <- build_tn_summary_plot(
    tn_summary_results
  )
  
  ggsave(
    filename =
      TN_FIGURE_OUTPUT_PATH,
    plot =
      tn_summary_plot,
    width =
      OUTPUT_WIDTH,
    height =
      OUTPUT_HEIGHT,
    units = "in",
    dpi =
      OUTPUT_DPI,
    device =
      grDevices::cairo_pdf,
    family = "Arial",
    limitsize = FALSE
  )
  
  message(
    paste0(
      "Saved TP per-release Hit and Over results to ",
      PER_RELEASE_OUTPUT_PATH
    )
  )
  
  message(
    paste0(
      "Saved TP summary statistics to ",
      SUMMARY_OUTPUT_PATH
    )
  )
  
  message(
    paste0(
      "Saved TP summary figure to ",
      FIGURE_OUTPUT_PATH
    )
  )
  
  message(
    paste0(
      "Saved TN per-release Hit and Over results to ",
      TN_PER_RELEASE_OUTPUT_PATH
    )
  )
  
  message(
    paste0(
      "Saved TN summary statistics to ",
      TN_SUMMARY_OUTPUT_PATH
    )
  )
  
  message(
    paste0(
      "Saved TN summary figure to ",
      TN_FIGURE_OUTPUT_PATH
    )
  )
}

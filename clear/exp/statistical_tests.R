
METRICS_ROOT <- "./metrics"
CLEAR_METRICS_ROOT <- file.path("..", "Data")
MODEL_NAME <- "logistic_regression"
RESULT_ROOT <- "./result_fig/statistical_tests"

DATASETS <- c("linedp_dataset", "glance_dataset")

CLEAR_FILE <- "CLEAR-SUM.csv"

BASELINE_FILES <- c(
  "SOUND-OP2" = "SOUND-Op2.csv",
  "GLANCE-LR" = "Glance-LR.csv",
  "LineDP" = "LineDP.csv",
  "DeepLineDP" = "DeepLineDP.csv",
  "N-gram" = "Ngram.csv",
  "ErrorProne" = "ErrorProne.csv"
)

METRIC_CONFIG <- list(
  "IFA" = list(aliases = c("ifa", "IFA"), direction = "lower"),
  "Recall@Top20%LOC" = list(aliases = c("recall_20", "Recall@20%LOC", "Recall@Top20%LOC"), direction = "higher"),
  "Effort@Top20%Recall" = list(aliases = c("effort@20%recall", "Effort@20%Recall", "effort_20"), direction = "lower"),
  "AUC" = list(aliases = c("auc", "AUC"), direction = "higher"),
  "D2H" = list(aliases = c("d2h", "D2H"), direction = "lower"),
  "FAR" = list(aliases = c("far", "FAR"), direction = "lower")
)

ALPHA <- 0.05
CLIFF_NEGLIGIBLE <- 0.147
CLIFF_SMALL <- 0.33
CLIFF_MEDIUM <- 0.474

normalize_name <- function(x) tolower(gsub("[^a-z0-9]", "", x))

find_column <- function(data, aliases, label) {
  normalized_names <- normalize_name(names(data))
  normalized_aliases <- normalize_name(aliases)
  matched <- match(normalized_aliases, normalized_names)
  matched <- matched[!is.na(matched)]
  if (length(matched) == 0) {
    stop(paste0("Cannot find ", label, " column. Available columns: ", paste(names(data), collapse = ", ")))
  }
  names(data)[matched[1]]
}

find_release_column <- function(data) {
  find_column(
    data,
    c("release", "test_release", "testrelease", "target_release", "targetrelease", "version"),
    "release"
  )
}

find_project_column <- function(data) {
  candidates <- c("project", "project_name", "projectname")
  normalized_names <- normalize_name(names(data))
  normalized_candidates <- normalize_name(candidates)
  matched <- match(normalized_candidates, normalized_names)
  matched <- matched[!is.na(matched)]
  if (length(matched) == 0) return(NULL)
  names(data)[matched[1]]
}

read_metric_file <- function(path) {
  if (!file.exists(path)) stop(paste0("File does not exist: ", path))
  read.csv(path, check.names = FALSE, stringsAsFactors = FALSE)
}

build_release_metric_table <- function(data, metric_aliases, value_name, source_name) {
  release_col <- find_release_column(data)
  project_col <- find_project_column(data)
  metric_col <- find_column(data, metric_aliases, "metric")
  
  release_value <- as.character(data[[release_col]])
  if (!is.null(project_col)) {
    key <- paste(as.character(data[[project_col]]), release_value, sep = "::")
  } else {
    key <- release_value
  }
  
  result <- data.frame(
    key = key,
    release = release_value,
    value = suppressWarnings(as.numeric(data[[metric_col]])),
    stringsAsFactors = FALSE
  )
  
  result <- result[is.finite(result$value), , drop = FALSE]
  
  duplicated_keys <- unique(result$key[duplicated(result$key)])
  if (length(duplicated_keys) > 0) {
    duplicated_rows <- result[result$key %in% duplicated_keys, , drop = FALSE]
    
    inconsistent_keys <- vapply(
      split(duplicated_rows$value, duplicated_rows$key),
      function(values) length(unique(values)) > 1,
      logical(1)
    )
    
    if (any(inconsistent_keys)) {
      bad_keys <- names(inconsistent_keys)[inconsistent_keys]
      stop(
        paste0(
          "Duplicated release keys with different metric values were found in ",
          source_name,
          ": ",
          paste(head(bad_keys, 20), collapse = ", "),
          if (length(bad_keys) > 20) " ..." else "",
          ". Please inspect these rows before statistical testing."
        )
      )
    }
    
    result <- result[!duplicated(result$key), , drop = FALSE]
  }
  
  names(result)[names(result) == "value"] <- value_name
  result
}

prepare_paired_values <- function(clear_data, baseline_data, metric_aliases, baseline_name) {
  clear_part <- build_release_metric_table(
    clear_data,
    metric_aliases,
    "clear_value",
    "CLEAR-SUM"
  )
  
  baseline_part <- build_release_metric_table(
    baseline_data,
    metric_aliases,
    "baseline_value",
    baseline_name
  )
  
  paired <- merge(
    clear_part[, c("key", "release", "clear_value")],
    baseline_part[, c("key", "baseline_value")],
    by = "key",
    all = FALSE,
    sort = FALSE
  )
  
  paired <- paired[
    is.finite(paired$clear_value) & is.finite(paired$baseline_value),
    ,
    drop = FALSE
  ]
  
  if (nrow(paired) == 0) stop(paste0("No valid paired release-level values were found for ", baseline_name, "."))
  paired
}

paired_wilcoxon_p <- function(x, y) {
  difference <- x - y
  if (all(difference == 0)) return(1)
  result <- suppressWarnings(
    wilcox.test(x, y, paired = TRUE, exact = FALSE, alternative = "two.sided")
  )
  if (is.finite(result$p.value)) result$p.value else 1
}

cliffs_delta <- function(x, y) {
  comparison <- outer(x, y, FUN = "-")
  (sum(comparison > 0) - sum(comparison < 0)) / (length(x) * length(y))
}

cliff_magnitude <- function(delta) {
  value <- abs(delta)
  if (value < CLIFF_NEGLIGIBLE) {
    "N"
  } else if (value < CLIFF_SMALL) {
    "S"
  } else if (value < CLIFF_MEDIUM) {
    "M"
  } else {
    "L"
  }
}

classify_wtl <- function(p_bh, delta, direction) {
  significant <- is.finite(p_bh) && p_bh < ALPHA
  non_negligible <- abs(delta) >= CLIFF_NEGLIGIBLE
  if (!significant || !non_negligible) return("T")
  clear_better <- if (direction == "higher") delta > 0 else delta < 0
  if (clear_better) "W" else "L"
}

format_p_value <- function(x) {
  if (!is.finite(x)) return("NA")
  if (x < 0.001) return("<0.001")
  sprintf("%.3f", x)
}

format_delta <- function(delta, magnitude) sprintf("%.3f(%s)", delta, magnitude)

latex_escape <- function(x) {
  x <- gsub("_", "\\\\_", x, fixed = TRUE)
  x <- gsub("%", "\\\\%", x, fixed = TRUE)
  x <- gsub("&", "\\\\&", x, fixed = TRUE)
  x
}

build_latex_tables <- function(dataset_name, detailed_results, summary_results) {
  baselines <- names(BASELINE_FILES)
  metrics <- names(METRIC_CONFIG)
  
  lines <- c(
    "% Required packages: booktabs, multirow",
    "",
    "\\begin{table*}[t]",
    paste0("\\caption{Statistical test results between CLEAR-SUM and the selected baselines on ", latex_escape(dataset_name), ".}"),
    paste0("\\label{tab:", dataset_name, "_statistical_tests}"),
    "\\centering",
    "\\scriptsize",
    paste0("\\begin{tabular}{ll", paste(rep("c", length(baselines)), collapse = ""), "}"),
    "\\toprule",
    paste(c("Metric", "Statistic", latex_escape(baselines)), collapse = " & "),
    "\\\\",
    "\\midrule"
  )
  
  for (metric_index in seq_along(metrics)) {
    metric <- metrics[metric_index]
    current <- detailed_results[detailed_results$Metric == metric, , drop = FALSE]
    current <- current[match(baselines, current$Baseline), , drop = FALSE]
    
    p_cells <- vapply(seq_len(nrow(current)), function(i) {
      value <- format_p_value(current$BH_P[i])
      if (current$BH_P[i] < ALPHA && abs(current$Cliff_Delta[i]) >= CLIFF_NEGLIGIBLE) {
        paste0("\\textbf{", value, "}")
      } else {
        value
      }
    }, character(1))
    
    delta_cells <- vapply(seq_len(nrow(current)), function(i) {
      value <- format_delta(current$Cliff_Delta[i], current$Magnitude[i])
      if (current$BH_P[i] < ALPHA && abs(current$Cliff_Delta[i]) >= CLIFF_NEGLIGIBLE) {
        paste0("\\textbf{", value, "}")
      } else {
        value
      }
    }, character(1))
    
    direction_symbol <- if (METRIC_CONFIG[[metric]]$direction == "higher") "$\\uparrow$" else "$\\downarrow$"
    
    lines <- c(
      lines,
      paste(
        c(
          paste0("\\multirow{2}{*}{", latex_escape(metric), " (", direction_symbol, ")}"),
          "BH-corrected $p$",
          p_cells
        ),
        collapse = " & "
      ),
      "\\\\",
      paste(c("", "Cliff's $\\delta$", delta_cells), collapse = " & "),
      "\\\\"
    )
    
    if (metric_index < length(metrics)) lines <- c(lines, "\\midrule")
  }
  
  lines <- c(
    lines,
    "\\bottomrule",
    "\\end{tabular}",
    "\\end{table*}",
    "",
    "\\begin{table}[t]",
    paste0("\\caption{W/T/L summary of CLEAR-SUM against the selected baselines on ", latex_escape(dataset_name), ".}"),
    paste0("\\label{tab:", dataset_name, "_wtl}"),
    "\\centering",
    "\\scriptsize",
    paste0("\\begin{tabular}{l", paste(rep("c", length(metrics)), collapse = ""), "}"),
    "\\toprule",
    paste(c("Method", latex_escape(metrics)), collapse = " & "),
    "\\\\",
    "\\midrule",
    paste(
      c(
        "CLEAR-SUM",
        vapply(
          metrics,
          function(metric) summary_results$WTL[summary_results$Metric == metric][1],
          character(1)
        )
      ),
      collapse = " & "
    ),
    "\\\\",
    "\\bottomrule",
    "\\end{tabular}",
    "\\end{table}"
  )
  
  lines
}

run_dataset_statistics <- function(dataset_name) {
  clear_path <- file.path(
    CLEAR_METRICS_ROOT,
    dataset_name,
    "results",
    "ranking_evaluation",
    MODEL_NAME,
    CLEAR_FILE
  )
  
  baseline_directory <- file.path(METRICS_ROOT, dataset_name)
  clear_data <- read_metric_file(clear_path)
  
  results <- data.frame(
    Dataset = character(),
    Metric = character(),
    Direction = character(),
    Baseline = character(),
    N_Pairs = integer(),
    Mean_CLEAR = numeric(),
    Mean_Baseline = numeric(),
    Median_CLEAR = numeric(),
    Median_Baseline = numeric(),
    Wilcoxon_P = numeric(),
    BH_P = numeric(),
    Cliff_Delta = numeric(),
    Magnitude = character(),
    Result = character(),
    stringsAsFactors = FALSE
  )
  
  for (metric_name in names(METRIC_CONFIG)) {
    metric_config <- METRIC_CONFIG[[metric_name]]
    metric_rows <- list()
    
    for (baseline_name in names(BASELINE_FILES)) {
      baseline_path <- file.path(baseline_directory, BASELINE_FILES[[baseline_name]])
      baseline_data <- read_metric_file(baseline_path)
      
      paired <- prepare_paired_values(clear_data, baseline_data, metric_config$aliases, baseline_name)
      
      p_raw <- paired_wilcoxon_p(paired$clear_value, paired$baseline_value)
      delta <- cliffs_delta(paired$clear_value, paired$baseline_value)
      
      metric_rows[[baseline_name]] <- data.frame(
        Dataset = dataset_name,
        Metric = metric_name,
        Direction = metric_config$direction,
        Baseline = baseline_name,
        N_Pairs = nrow(paired),
        Mean_CLEAR = mean(paired$clear_value),
        Mean_Baseline = mean(paired$baseline_value),
        Median_CLEAR = median(paired$clear_value),
        Median_Baseline = median(paired$baseline_value),
        Wilcoxon_P = p_raw,
        BH_P = NA_real_,
        Cliff_Delta = delta,
        Magnitude = cliff_magnitude(delta),
        Result = NA_character_,
        stringsAsFactors = FALSE
      )
    }
    
    metric_result <- do.call(rbind, metric_rows)
    metric_result$BH_P <- p.adjust(metric_result$Wilcoxon_P, method = "BH")
    metric_result$Result <- mapply(
      classify_wtl,
      metric_result$BH_P,
      metric_result$Cliff_Delta,
      metric_result$Direction,
      USE.NAMES = FALSE
    )
    
    results <- rbind(results, metric_result)
  }
  
  summary_results <- do.call(
    rbind,
    lapply(names(METRIC_CONFIG), function(metric_name) {
      current <- results[results$Metric == metric_name, , drop = FALSE]
      wins <- sum(current$Result == "W")
      ties <- sum(current$Result == "T")
      losses <- sum(current$Result == "L")
      data.frame(
        Dataset = dataset_name,
        Metric = metric_name,
        Wins = wins,
        Ties = ties,
        Losses = losses,
        WTL = paste0(wins, "/", ties, "/", losses),
        stringsAsFactors = FALSE
      )
    })
  )
  
  output_directory <- file.path(RESULT_ROOT, dataset_name)
  dir.create(output_directory, recursive = TRUE, showWarnings = FALSE)
  
  write.csv(
    results,
    file.path(output_directory, "CLEAR-SUM_statistical_tests.csv"),
    row.names = FALSE
  )
  
  write.csv(
    summary_results,
    file.path(output_directory, "CLEAR-SUM_WTL_summary.csv"),
    row.names = FALSE
  )
  
  writeLines(
    build_latex_tables(dataset_name, results, summary_results),
    file.path(output_directory, "CLEAR-SUM_statistical_tests_tables.tex")
  )
  
  cat("\nDataset:", dataset_name, "\n")
  print(
    results[, c("Metric", "Baseline", "N_Pairs", "BH_P", "Cliff_Delta", "Magnitude", "Result")],
    row.names = FALSE
  )
  cat("\nW/T/L summary:\n")
  print(summary_results[, c("Metric", "WTL")], row.names = FALSE)
}

for (dataset_name in DATASETS) {
  run_dataset_statistics(dataset_name)
}

suppressPackageStartupMessages({
  library(ggplot2)
  library(scales)
  library(ScottKnottESD)
})

METRICS_ROOT <- "./metrics"

CLEAR_METRICS_ROOT <- file.path(
  "..",
  "Data"
)

MODEL_NAME <- "logistic_regression"

RESULT_ROOT <- "./result_fig/RQ1"

DATASETS <- c(
  # "linedp_dataset"
 "glance_dataset"
)

METHOD_FILES <- c(
  "CLEAR-MAX.csv",
  "CLEAR-SUM.csv",
  "CLEAR-TOP2AVG.csv",
  "CLEAR-TOP3AVG.csv",
  "SOUND-Barinel.csv",
  "SOUND-Dstar.csv",
  "SOUND-Ochiai.csv",
  "SOUND-Op2.csv",
  "SOUND-Tarantula.csv",
  "DeepLineDP.csv",
  "GLANCE-LR.csv",
  "GLANCE-EA.csv",
  "GLANCE-MD.csv",
  "LineDP.csv",
  "ErrorProne.csv",
  "Ngram.csv"
)

METHOD_ORDER <- sub(
  "\\.csv$",
  "",
  METHOD_FILES
)

METHOD_COLORS <- setNames(
  ifelse(
    grepl("^CLEAR-", METHOD_ORDER),
    "#da936a",
    "#547da3"
  ),
  METHOD_ORDER
)

NON_METRIC_COLUMNS <- c(
  "release",
  "train_release",
  "test_release",
  "aggregation_method",
  "ranking_strategy",
  "previous_release_line_defect_rate",
  "project",
  "train",
  "test",
  "method",
  "group",
  "dataset",
  "filename"
)

TARGET_METRICS <- c(
  "recall_20",
  "effort_20_recall",
  "ifa",
  "auc",
  "d2h",
  "far"
)



normalize_metric_name <- function(metric_name)
{
  normalized_name <- tolower(
    gsub(
      "[^A-Za-z0-9]+",
      "_",
      metric_name
    )
  )
  
  normalized_name <- gsub(
    "^_+|_+$",
    "",
    normalized_name
  )
  
  return(normalized_name)
}


get_metric_display_name <- function(metric_name)
{
  normalized_name <- normalize_metric_name(
    metric_name
  )
  
  display_names <- c(
    "recall_20" = "Recall@Top20%LOC",
    "effort_20_recall" = "Effort@Top20%Recall",
    "ifa" = "IFA",
    "auc" = "AUC",
    "d2h" = "D2H",
    "far" = "FAR"
  )
  
  if (normalized_name %in% names(display_names)) {
    return(
      unname(
        display_names[[normalized_name]]
      )
    )
  }
  
  return(metric_name)
}


is_higher_better_metric <- function(metric_name)
{
  normalized_name <- normalize_metric_name(
    metric_name
  )
  
  lower_is_better_metrics <- c(
    "ifa",
    "far",
    "false_alarm_rate",
    "d2h",
    "distance_to_heaven",
    "effort",
    "effort_20_recall",
    "effort_at_20_recall",
    "fp",
    "fn",
    "loss"
  )
  
  return(
    !(normalized_name %in% lower_is_better_metrics)
  )
}


add_sk_esd_grouping <- function(
    metric_data,
    higher_is_better
)
{
  original_method_order <- levels(
    metric_data$group
  )
  
  grouped_values <- split(
    metric_data$value,
    as.character(metric_data$group)
  )
  
  grouped_values <- grouped_values[
    original_method_order
  ]
  
  group_lengths <- vapply(
    grouped_values,
    length,
    integer(1)
  )
  
  if (length(unique(group_lengths)) != 1) {
    stop(
      "Scott-Knott ESD requires the same number of observations for all methods."
    )
  }
  
  original_method_names <- names(
    grouped_values
  )
  safe_method_names <- make.names(
    original_method_names,
    unique = TRUE
  )
  names(grouped_values) <- safe_method_names
  
  sk_input <- as.data.frame(
    grouped_values,
    check.names = FALSE
  )
  
  all_values <- unlist(
    sk_input,
    use.names = FALSE
  )
  
  if (length(unique(all_values)) == 1) {
    sk_groups <- setNames(
      rep(1, ncol(sk_input)),
      colnames(sk_input)
    )
  } else {
    column_signatures <- vapply(
      sk_input,
      function(values) {
        paste(
          sprintf("%.17g", values),
          collapse = "\r"
        )
      },
      character(1)
    )
    
    representative_mask <- !duplicated(
      column_signatures
    )
    
    unique_sk_input <- sk_input[
      ,
      representative_mask,
      drop = FALSE
    ]
    
    if (ncol(unique_sk_input) == 1) {
      unique_sk_groups <- setNames(
        1,
        colnames(unique_sk_input)
      )
    } else {
      unique_sk_groups <- sk_esd(
        unique_sk_input
      )$group
      
      if (is.null(names(unique_sk_groups))) {
        names(unique_sk_groups) <- colnames(
          unique_sk_input
        )
      }
    }
    
    representative_signatures <- column_signatures[
      representative_mask
    ]
    
    signature_group_map <- setNames(
      as.numeric(
        unique_sk_groups[
          colnames(unique_sk_input)
        ]
      ),
      representative_signatures
    )
    
    sk_groups <- setNames(
      as.numeric(
        signature_group_map[
          column_signatures
        ]
      ),
      colnames(sk_input)
    )
  }
  
  if (higher_is_better) {
    ranking <- sk_groups
  } else {
    ranking <- (
      max(sk_groups) - sk_groups
    ) + 1
  }
  
  ranking_by_method <- setNames(
    as.numeric(
      ranking[
        safe_method_names
      ]
    ),
    original_method_names
  )
  
  ordered_methods <- original_method_order[
    order(
      ranking_by_method[
        original_method_order
      ],
      seq_along(original_method_order)
    )
  ]
  
  metric_data$group <- factor(
    as.character(metric_data$group),
    levels = ordered_methods
  )
  
  metric_data$sk_rank <- as.numeric(
    ranking_by_method[
      as.character(metric_data$group)
    ]
  )
  
  return(metric_data)
}


print_sk_esd_ranking <- function(
    metric_data,
    metric_name,
    dataset_name
)
{
  rank_data <- unique(
    data.frame(
      method = as.character(metric_data$group),
      rank = metric_data$sk_rank,
      stringsAsFactors = FALSE
    )
  )
  
  method_order <- levels(
    metric_data$group
  )
  
  rank_data$method_order <- match(
    rank_data$method,
    method_order
  )
  
  rank_data <- rank_data[
    order(
      rank_data$rank,
      rank_data$method_order
    ),
    ,
    drop = FALSE
  ]
  
  separator <- paste(
    rep("=", 80),
    collapse = ""
  )
  
  cat("\n", separator, "\n", sep = "")
  cat(
    "Dataset: ",
    dataset_name,
    " | Metric: ",
    get_metric_display_name(metric_name),
    "\n",
    sep = ""
  )
  cat(
    paste(
      rep("-", 80),
      collapse = ""
    ),
    "\n",
    sep = ""
  )
  
  for (rank_value in sort(unique(rank_data$rank))) {
    methods <- rank_data$method[
      rank_data$rank == rank_value
    ]
    
    cat(
      "Rank ",
      rank_value,
      ": ",
      paste(
        methods,
        collapse = ", "
      ),
      "\n",
      sep = ""
    )
  }
  
  cat(separator, "\n", sep = "")
}


create_output_filename <- function(metric_name)
{
  normalized_name <- normalize_metric_name(
    metric_name
  )
  
  if (normalized_name == "ifa") {
    return("RQ1_IFA.pdf")
  }
  
  if (normalized_name == "recall_20") {
    return("RQ1_recall.pdf")
  }
  
  if (
    normalized_name %in% c(
      "effort",
      "effort_20_recall"
    )
  ) {
    return("RQ1_effort.pdf")
  }
  
  return(
    paste0(
      "RQ1_",
      gsub(
        "[^A-Za-z0-9]+",
        "_",
        metric_name
      ),
      ".pdf"
    )
  )
}


create_metric_spec <- function(metric_name)
{
  normalized_name <- normalize_metric_name(
    metric_name
  )
  higher_is_better <- is_higher_better_metric(
    metric_name
  )
  
  if (normalized_name == "ifa") {
    return(
      list(
        name = metric_name,
        output = create_output_filename(
          metric_name
        ),
        scale_type = "log",
        lower_bound = 1,
        upper_bound = 10000,
        label_accuracy = 1,
        higher_is_better = higher_is_better,
        transform = function(values) {
          pmin(
            pmax(values + 1, 1),
            10000
          )
        }
      )
    )
  }
  
  bounded_zero_one_metrics <- c(
    "accuracy",
    "acc",
    "precision",
    "recall",
    "far",
    "d2h",
    "f1",
    "auc",
    "effort",
    "effort_20_recall"
  )
  
  is_recall_percentage <- grepl(
    "^recall_[0-9]+$",
    normalized_name
  )
  
  if (
    normalized_name %in% bounded_zero_one_metrics ||
    is_recall_percentage
  ) {
    return(
      list(
        name = metric_name,
        output = create_output_filename(
          metric_name
        ),
        scale_type = "linear",
        lower_bound = 0,
        upper_bound = 1,
        label_accuracy = if (normalized_name == "far") {
          0.0001
        } else {
          0.01
        },
        higher_is_better = higher_is_better,
        transform = function(values) {
          pmin(
            pmax(values, 0),
            1
          )
        }
      )
    )
  }
  
  if (normalized_name == "mcc") {
    return(
      list(
        name = metric_name,
        output = create_output_filename(
          metric_name
        ),
        scale_type = "linear",
        lower_bound = -1,
        upper_bound = 1,
        label_accuracy = 0.01,
        higher_is_better = higher_is_better,
        transform = function(values) {
          pmin(
            pmax(values, -1),
            1
          )
        }
      )
    )
  }
  
  nonnegative_metrics <- c(
    "tp",
    "fp",
    "fn",
    "tn",
    "total_lines",
    "total_defective_lines"
  )
  
  if (normalized_name %in% nonnegative_metrics) {
    return(
      list(
        name = metric_name,
        output = create_output_filename(
          metric_name
        ),
        scale_type = "linear",
        lower_bound = 0,
        upper_bound = Inf,
        label_accuracy = 1,
        higher_is_better = higher_is_better,
        transform = function(values) {
          pmax(values, 0)
        }
      )
    )
  }
  
  return(
    list(
      name = metric_name,
      output = create_output_filename(
        metric_name
      ),
      scale_type = "linear",
      lower_bound = -Inf,
      upper_bound = Inf,
      label_accuracy = 0.01,
      higher_is_better = higher_is_better,
      transform = identity
    )
  )
}


discover_metric_names <- function(
    dataset_directory,
    dataset_name
)
{
  metric_names <- character()
  
  for (file_name in METHOD_FILES) {
    if (grepl("^CLEAR-", file_name)) {
      file_path <- file.path(
        CLEAR_METRICS_ROOT,
        dataset_name,
        "results",
        "ranking_evaluation",
        MODEL_NAME,
        file_name
      )
    } else {
      file_path <- file.path(
        dataset_directory,
        file_name
      )
    }
    
    if (!file.exists(file_path)) {
      next
    }
    
    current_data <- read.csv(
      file_path,
      check.names = FALSE,
      stringsAsFactors = FALSE
    )
    
    candidate_columns <- setdiff(
      names(current_data),
      NON_METRIC_COLUMNS
    )
    
    for (column_name in candidate_columns) {
      numeric_values <- suppressWarnings(
        as.numeric(
          current_data[[column_name]]
        )
      )
      
      if (any(is.finite(numeric_values))) {
        if (!(column_name %in% metric_names)) {
          metric_names <- c(
            metric_names,
            column_name
          )
        }
      }
    }
  }
  
  normalized_metric_names <- vapply(
    metric_names,
    normalize_metric_name,
    character(1)
  )
  
  missing_metrics <- TARGET_METRICS[
    !(TARGET_METRICS %in% normalized_metric_names)
  ]
  
  if (length(missing_metrics) > 0) {
    stop(
      paste0(
        "The following required metrics were not found in ",
        dataset_directory,
        ": ",
        paste(
          missing_metrics,
          collapse = ", "
        ),
        "."
      )
    )
  }
  
  metric_names <- metric_names[
    match(
      TARGET_METRICS,
      normalized_metric_names
    )
  ]
  
  return(metric_names)
}


load_metric_data <- function(
    dataset_directory,
    dataset_name,
    metric_spec
)
{
  metric_data <- data.frame(
    value = numeric(),
    group = character(),
    stringsAsFactors = FALSE
  )
  
  for (file_name in METHOD_FILES) {
    if (grepl("^CLEAR-", file_name)) {
      file_path <- file.path(
        CLEAR_METRICS_ROOT,
        dataset_name,
        "results",
        "ranking_evaluation",
        MODEL_NAME,
        file_name
      )
    } else {
      file_path <- file.path(
        dataset_directory,
        file_name
      )
    }
    
    if (!file.exists(file_path)) {
      next
    }
    
    current_data <- read.csv(
      file_path,
      check.names = FALSE,
      stringsAsFactors = FALSE
    )
    
    metric_column <- if (
      normalize_metric_name(metric_spec$name) == "effort_20_recall"
    ) {
      "effort@20%recall"
    } else {
      metric_spec$name
    }
    
    if (!(metric_column %in% names(current_data))) {
      next
    }
    
    metric_values <- suppressWarnings(
      as.numeric(
        current_data[[metric_column]]
      )
    )
    
    transformed_values <- metric_spec$transform(
      metric_values
    )
    
    current_metric_data <- data.frame(
      value = transformed_values,
      group = sub(
        "\\.csv$",
        "",
        file_name
      ),
      stringsAsFactors = FALSE
    )
    
    current_metric_data <- current_metric_data[
      is.finite(current_metric_data$value),
      ,
      drop = FALSE
    ]
    
    metric_data <- rbind(
      metric_data,
      current_metric_data
    )
  }
  
  if (nrow(metric_data) == 0) {
    stop(
      paste0(
        "No valid data were found for metric ",
        metric_spec$name,
        " in ",
        dataset_directory,
        "."
      )
    )
  }
  
  group_medians <- aggregate(
    value ~ group,
    data = metric_data,
    FUN = mean,
    na.rm = TRUE
  )
  
  group_medians$original_order <- match(
    group_medians$group,
    METHOD_ORDER
  )
  
  if (metric_spec$higher_is_better) {
    group_medians <- group_medians[
      order(
        -group_medians$value,
        group_medians$original_order
      ),
      ,
      drop = FALSE
    ]
  } else {
    group_medians <- group_medians[
      order(
        group_medians$value,
        group_medians$original_order
      ),
      ,
      drop = FALSE
    ]
  }
  
  sorted_methods <- as.character(
    group_medians$group
  )
  
  metric_data$group <- factor(
    metric_data$group,
    levels = sorted_methods
  )
  
  return(metric_data)
}


calculate_adaptive_limits <- function(
    metric_data,
    metric_spec,
    dataset_name
)
{
  normalized_name <- normalize_metric_name(
    metric_spec$name
  )
  
  if (normalized_name == "ifa") {
    return(
      c(
        1,
        10000
      )
    )
  }
  
  if (normalized_name == "far") {
    if (dataset_name == "glance_dataset") {
      return(
        c(
          0.180,
          0.205
        )
      )
    }
    
    return(
      c(
        0.1978,
        0.2007
      )
    )
  }
  
  grouped_values <- split(
    metric_data$value,
    metric_data$group
  )
  
  visible_ranges <- lapply(
    grouped_values,
    function(values) {
      values <- values[
        is.finite(values)
      ]
      
      if (length(values) == 0) {
        return(
          c(
            NA_real_,
            NA_real_
          )
        )
      }
      
      boxplot_statistics <- boxplot.stats(
        values
      )$stats
      
      group_mean <- mean(
        values,
        na.rm = TRUE
      )
      
      return(
        c(
          min(
            boxplot_statistics,
            group_mean,
            na.rm = TRUE
          ),
          max(
            boxplot_statistics,
            group_mean,
            na.rm = TRUE
          )
        )
      )
    }
  )
  
  minimum_value <- min(
    vapply(
      visible_ranges,
      function(value_range) {
        value_range[[1]]
      },
      numeric(1)
    ),
    na.rm = TRUE
  )
  
  maximum_value <- max(
    vapply(
      visible_ranges,
      function(value_range) {
        value_range[[2]]
      },
      numeric(1)
    ),
    na.rm = TRUE
  )
  
  lower_limit <- minimum_value - 0.05
  upper_limit <- maximum_value + 0.05
  
  if (is.finite(metric_spec$lower_bound)) {
    lower_limit <- max(
      lower_limit,
      metric_spec$lower_bound
    )
  }
  
  if (is.finite(metric_spec$upper_bound)) {
    upper_limit <- min(
      upper_limit,
      metric_spec$upper_bound
    )
  }
  
  if (lower_limit >= upper_limit) {
    if (metric_spec$scale_type == "log") {
      lower_limit <- max(
        metric_spec$lower_bound,
        minimum_value / 1.05
      )
      upper_limit <- min(
        metric_spec$upper_bound,
        maximum_value * 1.05
      )
    } else {
      lower_limit <- minimum_value - 0.05
      upper_limit <- maximum_value + 0.05
      
      if (is.finite(metric_spec$lower_bound)) {
        lower_limit <- max(
          lower_limit,
          metric_spec$lower_bound
        )
      }
      
      if (is.finite(metric_spec$upper_bound)) {
        upper_limit <- min(
          upper_limit,
          metric_spec$upper_bound
        )
      }
    }
  }
  
  if (lower_limit >= upper_limit) {
    if (is.finite(metric_spec$lower_bound)) {
      lower_limit <- metric_spec$lower_bound
    } else {
      lower_limit <- minimum_value - 0.05
    }
    
    if (is.finite(metric_spec$upper_bound)) {
      upper_limit <- metric_spec$upper_bound
    } else {
      upper_limit <- maximum_value + 0.05
    }
  }
  
  return(
    c(
      lower_limit,
      upper_limit
    )
  )
}


build_metric_plot <- function(
    metric_data,
    metric_spec,
    dataset_name
)
{
  available_methods <- levels(
    metric_data$group
  )
  
  adaptive_limits <- calculate_adaptive_limits(
    metric_data = metric_data,
    metric_spec = metric_spec,
    dataset_name = dataset_name
  )
  
  group_means <- aggregate(
    value ~ group,
    data = metric_data,
    FUN = mean,
    na.rm = TRUE
  )
  
  group_means$group <- factor(
    group_means$group,
    levels = available_methods
  )
  
  method_rank_data <- unique(
    data.frame(
      group = as.character(metric_data$group),
      sk_rank = metric_data$sk_rank,
      stringsAsFactors = FALSE
    )
  )
  
  method_rank_data$group <- factor(
    method_rank_data$group,
    levels = available_methods
  )
  
  group_means <- merge(
    group_means,
    method_rank_data,
    by = "group",
    all.x = TRUE,
    sort = FALSE
  )
  
  rank_values <- sort(
    unique(metric_data$sk_rank)
  )
  
  metric_data$rank_panel <- factor(
    metric_data$sk_rank,
    levels = rank_values,
    labels = paste("Rank", rank_values)
  )
  
  group_means$rank_panel <- factor(
    group_means$sk_rank,
    levels = rank_values,
    labels = paste("Rank", rank_values)
  )
  
  plot_object <- ggplot(
    metric_data,
    aes(
      x = group,
      y = value,
      fill = group
    )
  ) +
    geom_boxplot(
      outlier.shape = NA,
      na.rm = TRUE
    ) +
    geom_point(
      data = group_means,
      aes(
        x = group,
        y = value
      ),
      inherit.aes = FALSE,
      shape = 23,
      size = 2,
      fill = "white",
      color = "black",
      na.rm = TRUE
    ) +
    scale_fill_manual(
      values = METHOD_COLORS[
        available_methods
      ],
      drop = FALSE
    ) +
    scale_x_discrete(
      drop = TRUE
    ) +
    facet_grid(
      . ~ rank_panel,
      scales = "free_x",
      space = "free_x"
    ) +
    guides(fill = "none") +
    labs(
      x = "",
      y = "",
      title = ""
    ) +
    theme_minimal() +
    theme(
      plot.background = element_rect(
        fill = "white",
        color = NA
      ),
      panel.background = element_rect(
        fill = "white",
        color = NA
      ),
      panel.grid.major = element_line(
        color = "#D9D9D9",
        linewidth = 0.5
      ),
      panel.grid.minor = element_line(
        color = "#EEEEEE",
        linewidth = 0.3
      ),
      panel.spacing.x = unit(
        0.35,
        "lines"
      ),
      strip.background = element_rect(
        fill = "#D9D9D9",
        color = NA
      ),
      strip.text.x = element_text(
        size = 18,
        face = "bold",
        color = "black"
      ),
      axis.text.x = element_text(
        size = 24,
        angle = 30,
        hjust = 0.75,
        vjust = 0.75,
        color = "black"
      ),
      axis.title.y = element_text(
        size = 14,
        color = "black"
      ),
      axis.text.y = element_text(
        size = 24,
        color = "black"
      ),
      plot.margin = margin(
        t = 12,
        r = 18,
        b = 35,
        l = 55
      )
    )
  
  if (metric_spec$scale_type == "log") {
    log_breaks <- c(
      1,
      10,
      100,
      1000
    )
    
    log_breaks <- log_breaks[
      log_breaks >= adaptive_limits[[1]] &
        log_breaks <= adaptive_limits[[2]]
    ]
    
    plot_object <- plot_object +
      scale_y_log10(
        breaks = log_breaks,
        labels = trans_format(
          "log10",
          math_format(10^.x)
        ),
        expand = expansion(
          mult = c(0, 0)
        )
      ) +
      coord_cartesian(
        ylim = adaptive_limits,
        clip = "on"
      )
  } else {
    axis_breaks <- pretty(
      adaptive_limits,
      n = 6
    )
    
    axis_breaks <- axis_breaks[
      axis_breaks >= adaptive_limits[[1]] &
        axis_breaks <= adaptive_limits[[2]]
    ]
    
    plot_object <- plot_object +
      scale_y_continuous(
        breaks = axis_breaks,
        labels = label_number(
          accuracy = metric_spec$label_accuracy,
          big.mark = ","
        ),
        expand = expansion(
          mult = c(0, 0)
        )
      ) +
      coord_cartesian(
        ylim = adaptive_limits,
        clip = "on"
      )
  }
  
  return(plot_object)
}


save_metric_plot <- function(
    plot_object,
    output_path,
    method_count
)
{
  plot_width <- max(
    16,
    method_count * 1.10
  )
  
  ggsave(
    filename = output_path,
    plot = plot_object,
    width = plot_width,
    height = 8,
    units = "in",
    dpi = 200,
    limitsize = FALSE
  )
}


for (dataset_name in DATASETS) {
  dataset_directory <- file.path(
    METRICS_ROOT,
    dataset_name
  )
  
  if (!dir.exists(dataset_directory)) {
    stop(
      paste0(
        "Dataset metric directory does not exist: ",
        dataset_directory
      )
    )
  }
  
  dataset_result_directory <- file.path(
    RESULT_ROOT,
    dataset_name
  )
  
  dir.create(
    dataset_result_directory,
    recursive = TRUE,
    showWarnings = FALSE
  )
  
  metric_names <- discover_metric_names(
    dataset_directory,
    dataset_name
  )
  
  metric_specs <- lapply(
    metric_names,
    create_metric_spec
  )
  
  for (metric_spec in metric_specs) {
    metric_data <- load_metric_data(
      dataset_directory = dataset_directory,
      dataset_name = dataset_name,
      metric_spec = metric_spec
    )
    
    metric_data <- add_sk_esd_grouping(
      metric_data = metric_data,
      higher_is_better = metric_spec$higher_is_better
    )
    
    print_sk_esd_ranking(
      metric_data = metric_data,
      metric_name = metric_spec$name,
      dataset_name = dataset_name
    )
    
    plot_object <- build_metric_plot(
      metric_data = metric_data,
      metric_spec = metric_spec,
      dataset_name = dataset_name
    )
    
    output_path <- file.path(
      dataset_result_directory,
      metric_spec$output
    )
    
    save_metric_plot(
      plot_object = plot_object,
      output_path = output_path,
      method_count = nlevels(
        metric_data$group
      )
    )
  }
}


# Export the mean and median of every available metric for each method.
save_all_metrics_mean_median <- function(
    dataset_directory,
    dataset_name,
    output_path
)
{
  metric_names <- discover_metric_names(
    dataset_directory,
    dataset_name
  )
  
  summary_rows <- list()
  row_index <- 1
  
  for (file_name in METHOD_FILES) {
    if (grepl("^CLEAR-", file_name)) {
      file_path <- file.path(
        CLEAR_METRICS_ROOT,
        dataset_name,
        "results",
        "ranking_evaluation",
        MODEL_NAME,
        file_name
      )
    } else {
      file_path <- file.path(
        dataset_directory,
        file_name
      )
    }
    
    if (!file.exists(file_path)) {
      next
    }
    
    current_data <- read.csv(
      file_path,
      check.names = FALSE,
      stringsAsFactors = FALSE
    )
    
    summary_row <- data.frame(
      Method = sub(
        "\\.csv$",
        "",
        file_name
      ),
      stringsAsFactors = FALSE,
      check.names = FALSE
    )
    
    for (metric_name in metric_names) {
      metric_column <- metric_name
      
      if (!(metric_column %in% names(current_data))) {
        if (
          normalize_metric_name(metric_name) == "effort_20_recall" &&
          "effort@20%recall" %in% names(current_data)
        ) {
          metric_column <- "effort@20%recall"
        } else {
          summary_row[[paste0(metric_name, "_Mean")]] <- NA_real_
          summary_row[[paste0(metric_name, "_Median")]] <- NA_real_
          next
        }
      }
      
      metric_values <- suppressWarnings(
        as.numeric(
          current_data[[metric_column]]
        )
      )
      metric_values <- metric_values[
        is.finite(metric_values)
      ]
      
      summary_row[[paste0(metric_name, "_Mean")]] <- if (
        length(metric_values) > 0
      ) {
        mean(metric_values)
      } else {
        NA_real_
      }
      
      summary_row[[paste0(metric_name, "_Median")]] <- if (
        length(metric_values) > 0
      ) {
        median(metric_values)
      } else {
        NA_real_
      }
    }
    
    summary_rows[[row_index]] <- summary_row
    row_index <- row_index + 1
  }
  
  if (length(summary_rows) == 0) {
    stop(
      paste0(
        "No method data were available for summary in ",
        dataset_directory,
        "."
      )
    )
  }
  
  summary_data <- do.call(
    rbind,
    summary_rows
  )
  
  write.csv(
    summary_data,
    output_path,
    row.names = FALSE,
    na = ""
  )
  
  print(
    paste0(
      "Saved all-metric mean/median summary to ",
      output_path
    )
  )
}


for (dataset_name in DATASETS) {
  dataset_directory <- file.path(
    METRICS_ROOT,
    dataset_name
  )
  
  dataset_result_directory <- file.path(
    RESULT_ROOT,
    dataset_name
  )
  
  dir.create(
    dataset_result_directory,
    recursive = TRUE,
    showWarnings = FALSE
  )
  
  summary_output_path <- file.path(
    dataset_result_directory,
    "RQ1_all_metrics_mean_median.csv"
  )
  
  save_all_metrics_mean_median(
    dataset_directory = dataset_directory,
    dataset_name = dataset_name,
    output_path = summary_output_path
  )
}

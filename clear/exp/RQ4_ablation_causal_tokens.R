suppressPackageStartupMessages({
  library(ggplot2)
  library(scales)
})

SUMMARY_STATISTIC <- "mean"
# SUMMARY_STATISTIC <- "median"

AGGREGATION_METHODS <- c(
  "SUM",
  "MAX",
  "TOP2AVG",
  "TOP3AVG"
)

DATA_ROOT <- file.path(
  "..",
  "Data"
)

DATASET_NAME <- "linedp_dataset"
MODEL_NAME <- "logistic_regression"

RESULT_ROOT <- file.path(
  DATA_ROOT,
  DATASET_NAME,
  "results"
)

STRATEGY_ROOTS <- c(
  "clear" = file.path(
    RESULT_ROOT,
    "ranking_evaluation",
    MODEL_NAME
  ),
  "without_file_causal_token" = file.path(
    RESULT_ROOT,
    "ranking_evaluation_without_file_counterfactual_token",
    MODEL_NAME
  ),
  "without_line_causal_token" = file.path(
    RESULT_ROOT,
    "ranking_evaluation_without_line_counterfactual_token",
    MODEL_NAME
  ),
  "without_line_and_file_causal_tokens" = file.path(
    RESULT_ROOT,
    "ranking_evaluation_without_line_and_file_counterfactual_tokens",
    MODEL_NAME
  )
)

BASE_OUTPUT_ROOT <- file.path(
  ".",
  "result_fig",
  "RQ_ablation_ranking_causal_tokens",
  DATASET_NAME
)

STRATEGY_ORDER <- c(
  "clear",
  "without_file_causal_token",
  "without_line_causal_token",
  "without_line_and_file_causal_tokens"
)

STRATEGY_LABELS <- c(
  "clear" =
    "CLEAR",
  "without_file_causal_token" =
    "w/o File-Level Causal Tokens",
  "without_line_causal_token" =
    "w/o Line-Level Causal Tokens",
  "without_line_and_file_causal_tokens" =
    "w/o Both Causal Tokens"
)

STRATEGY_COLORS <- c(
  "clear" =
    "#B85C38",
  "without_file_causal_token" =
    "#D99563",
  "without_line_causal_token" =
    "#6F8FAF",
  "without_line_and_file_causal_tokens" =
    "#A7ADB4"
)

METRIC_ORDER <- c(
  "recall_20",
  "effort@20%recall",
  "far",
  "auc",
  "d2h",
  "ifa"
)

METRIC_LABELS <- c(
  "recall_20" = "Recall@20% (\u2191)",
  "effort@20%recall" = "Effort@20% (\u2193)",
  "far" = "FAR (\u2193)",
  "auc" = "AUC (\u2191)",
  "d2h" = "D2H (\u2193)",
  "ifa" = "IFA (\u2193)"
)

LEFT_Y_LIMITS <- c(
  0,
  0.7
)

IFA_RIGHT_LIMITS <- c(
  10,
  1000
)

IFA_RIGHT_PADDING_RATIO <- 0.10
LEFT_AXIS_ACCURACY <- 0.01
BAR_LABEL_ACCURACY <- 0.00001
IFA_BAR_LABEL_ACCURACY <- 1
BAR_WIDTH <- 0.72
DODGE_WIDTH <- 0.82
OUTPUT_WIDTH <- 16
OUTPUT_HEIGHT <- 8
OUTPUT_DPI <- 300


validate_configuration <- function()
{
  valid_statistics <- c(
    "mean",
    "median"
  )

  if (!(SUMMARY_STATISTIC %in% valid_statistics)) {
    stop(
      paste0(
        "SUMMARY_STATISTIC must be one of: ",
        paste(
          valid_statistics,
          collapse = ", "
        ),
        "."
      )
    )
  }

  if (length(AGGREGATION_METHODS) != 4) {
    stop(
      "Exactly four aggregation methods must be configured."
    )
  }
}


get_input_filename <- function(
    aggregation_method
)
{
  paste0(
    "CLEAR-",
    aggregation_method,
    ".csv"
  )
}


load_metric_file <- function(
    input_path,
    strategy_name
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

  input_data <- read.csv(
    input_path,
    check.names = FALSE,
    stringsAsFactors = FALSE
  )

  required_columns <- c(
    "release",
    METRIC_ORDER
  )

  missing_columns <- setdiff(
    required_columns,
    names(input_data)
  )

  if (length(missing_columns) > 0) {
    stop(
      paste0(
        "The input file is missing the following columns: ",
        paste(
          missing_columns,
          collapse = ", "
        ),
        ". File: ",
        input_path
      )
    )
  }

  input_data <- input_data[
    ,
    required_columns,
    drop = FALSE
  ]

  input_data$release <- as.character(
    input_data$release
  )

  if (any(duplicated(input_data$release))) {
    stop(
      paste0(
        "Duplicate release rows were found in ",
        input_path,
        "."
      )
    )
  }

  for (metric_name in METRIC_ORDER) {
    input_data[[metric_name]] <- suppressWarnings(
      as.numeric(
        input_data[[metric_name]]
      )
    )
  }

  input_data$ranking_strategy <- strategy_name

  input_data
}


load_ablation_data <- function(
    aggregation_method
)
{
  input_filename <- get_input_filename(
    aggregation_method
  )

  strategy_data <- lapply(
    STRATEGY_ORDER,
    function(strategy_name) {
      load_metric_file(
        file.path(
          STRATEGY_ROOTS[[strategy_name]],
          input_filename
        ),
        strategy_name
      )
    }
  )

  reference_releases <- strategy_data[[1]]$release

  for (data_index in seq_along(strategy_data)) {
    if (!setequal(
      reference_releases,
      strategy_data[[data_index]]$release
    )) {
      stop(
        paste0(
          "Release sets do not match for aggregation method ",
          aggregation_method,
          "."
        )
      )
    }
  }

  strategy_data <- lapply(
    strategy_data,
    function(input_data) {
      input_data[
        match(
          reference_releases,
          input_data$release
        ),
        ,
        drop = FALSE
      ]
    }
  )

  combined_data <- do.call(
    rbind,
    strategy_data
  )

  combined_data$ranking_strategy <- factor(
    combined_data$ranking_strategy,
    levels = STRATEGY_ORDER
  )

  combined_data
}


get_selected_summary_values <- function(
    summary_data
)
{
  if (SUMMARY_STATISTIC == "mean") {
    return(
      summary_data$mean_value
    )
  }

  summary_data$median_value
}


summarize_all_releases <- function(
    input_data
)
{
  summary_rows <- list()
  row_index <- 1

  for (metric_name in METRIC_ORDER) {
    metric_values <- suppressWarnings(
      as.numeric(
        input_data[[metric_name]]
      )
    )

    for (strategy_name in STRATEGY_ORDER) {
      strategy_mask <- as.character(
        input_data$ranking_strategy
      ) == strategy_name

      values <- metric_values[
        strategy_mask &
          is.finite(metric_values)
      ]

      if (length(values) == 0) {
        stop(
          paste0(
            "No finite values were found for metric ",
            metric_name,
            " and strategy ",
            strategy_name,
            "."
          )
        )
      }

      summary_rows[[row_index]] <- data.frame(
        metric = metric_name,
        ranking_strategy = strategy_name,
        mean_value = mean(values),
        median_value = median(values),
        standard_deviation = if (
          length(values) > 1
        ) {
          sd(values)
        } else {
          0
        },
        release_count = length(values),
        stringsAsFactors = FALSE
      )

      row_index <- row_index + 1
    }
  }

  summary_data <- do.call(
    rbind,
    summary_rows
  )

  summary_data$metric <- factor(
    summary_data$metric,
    levels = METRIC_ORDER
  )

  summary_data$ranking_strategy <- factor(
    summary_data$ranking_strategy,
    levels = STRATEGY_ORDER
  )

  summary_data <- summary_data[
    order(
      summary_data$metric,
      summary_data$ranking_strategy
    ),
    ,
    drop = FALSE
  ]

  summary_data
}


calculate_ifa_limits <- function(
    summary_data
)
{
  if (!is.null(IFA_RIGHT_LIMITS)) {
    return(
      IFA_RIGHT_LIMITS
    )
  }

  selected_summary_values <- get_selected_summary_values(
    summary_data
  )

  ifa_values <- selected_summary_values[
    as.character(
      summary_data$metric
    ) == "ifa"
  ]

  ifa_values <- ifa_values[
    is.finite(ifa_values) &
      ifa_values > 0
  ]

  if (length(ifa_values) == 0) {
    stop(
      paste0(
        "No finite positive IFA ",
        SUMMARY_STATISTIC,
        " values were found."
      )
    )
  }

  minimum_exponent <- floor(
    log10(
      min(
        ifa_values
      )
    )
  )

  maximum_exponent <- ceiling(
    log10(
      max(
        ifa_values
      ) * (
        1 + IFA_RIGHT_PADDING_RATIO
      )
    )
  )

  if (minimum_exponent >= maximum_exponent) {
    maximum_exponent <- minimum_exponent + 1
  }

  c(
    10 ^ minimum_exponent,
    10 ^ maximum_exponent
  )
}


transform_ifa_to_left_axis <- function(
    values,
    ifa_limits
)
{
  log_ifa_limits <- log10(
    ifa_limits
  )

  ifa_range <- diff(
    log_ifa_limits
  )

  left_range <- diff(
    LEFT_Y_LIMITS
  )

  if (
    any(
      !is.finite(values) |
        values <= 0
    ) ||
      !is.finite(ifa_range) ||
      ifa_range <= 0
  ) {
    stop(
      "IFA values and right-axis limits must be positive."
    )
  }

  transformed_values <- (
    log10(
      values
    ) - log_ifa_limits[1]
  ) / ifa_range

  (
    transformed_values * left_range
  ) + LEFT_Y_LIMITS[1]
}


prepare_plot_data <- function(
    summary_data
)
{
  ifa_limits <- calculate_ifa_limits(
    summary_data
  )

  plot_data <- summary_data

  plot_data$selected_value <- get_selected_summary_values(
    plot_data
  )

  plot_data$summary_statistic <- SUMMARY_STATISTIC
  plot_data$plot_value <- plot_data$selected_value

  ifa_mask <- as.character(
    plot_data$metric
  ) == "ifa"

  plot_data$plot_value[
    ifa_mask
  ] <- transform_ifa_to_left_axis(
    values = plot_data$selected_value[
      ifa_mask
    ],
    ifa_limits = ifa_limits
  )

  plot_data$value_label <- label_number(
    accuracy = BAR_LABEL_ACCURACY,
    big.mark = ","
  )(
    plot_data$selected_value
  )

  plot_data$value_label[
    ifa_mask
  ] <- label_number(
    accuracy = IFA_BAR_LABEL_ACCURACY,
    big.mark = ","
  )(
    plot_data$selected_value[
      ifa_mask
    ]
  )

  list(
    plot_data = plot_data,
    ifa_limits = ifa_limits
  )
}


build_combined_barplot <- function(
    plot_data,
    ifa_limits
)
{
  left_range <- diff(
    LEFT_Y_LIMITS
  )

  log_ifa_limits <- log10(
    ifa_limits
  )

  ifa_range <- diff(
    log_ifa_limits
  )

  ifa_group_index <- match(
    "ifa",
    METRIC_ORDER
  )

  ggplot(
    plot_data,
    aes(
      x = metric,
      y = plot_value,
      fill = ranking_strategy
    )
  ) +
    annotate(
      "rect",
      xmin = ifa_group_index - 0.5,
      xmax = ifa_group_index + 0.5,
      ymin = -Inf,
      ymax = Inf,
      fill = "#F4F4F4",
      alpha = 0.75
    ) +
    geom_vline(
      xintercept = c(
        ifa_group_index - 0.5,
        ifa_group_index + 0.5
      ),
      color = "#C8C8C8",
      linewidth = 0.45,
      linetype = "dashed"
    ) +
    geom_col(
      position = position_dodge(
        width = DODGE_WIDTH
      ),
      width = BAR_WIDTH,
      color = "#3F3F3F",
      linewidth = 0.35,
      na.rm = TRUE
    ) +
    geom_text(
      aes(
        label = value_label
      ),
      position = position_dodge(
        width = DODGE_WIDTH
      ),
      angle = 90,
      hjust = -0.06,
      vjust = 0.5,
      size = 3.7,
      color = "black",
      na.rm = TRUE
    ) +
    scale_fill_manual(
      values = STRATEGY_COLORS,
      breaks = STRATEGY_ORDER,
      labels = STRATEGY_LABELS[
        STRATEGY_ORDER
      ],
      drop = FALSE
    ) +
    scale_x_discrete(
      limits = METRIC_ORDER,
      labels = METRIC_LABELS[
        METRIC_ORDER
      ],
      drop = FALSE
    ) +
    scale_y_continuous(
      breaks = seq(
        LEFT_Y_LIMITS[1],
        LEFT_Y_LIMITS[2],
        by = 0.10
      ),
      labels = label_number(
        accuracy = LEFT_AXIS_ACCURACY
      ),
      sec.axis = sec_axis(
        trans = ~ 10 ^ (
          (
            (
              . - LEFT_Y_LIMITS[1]
            ) / left_range
          ) * ifa_range + log_ifa_limits[1]
        ),
        name = "IFA",
        breaks = 10 ^ seq(
          log_ifa_limits[1],
          log_ifa_limits[2],
          by = 1
        ),
        labels = trans_format(
          "log10",
          math_format(
            10 ^ .x
          )
        )
      ),
      expand = expansion(
        mult = c(
          0,
          0.025
        )
      )
    ) +
    coord_cartesian(
      ylim = LEFT_Y_LIMITS,
      clip = "off"
    ) +
    guides(
      fill = guide_legend(
        title = NULL,
        nrow = 1,
        byrow = TRUE
      )
    ) +
    labs(
      x = NULL,
      y = "Metric Value",
      title = ""
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
      panel.grid.major.x = element_blank(),
      panel.grid.minor.x = element_blank(),
      panel.grid.major.y = element_line(
        color = "#D7D7D7",
        linewidth = 0.50
      ),
      panel.grid.minor.y = element_blank(),
      axis.title.x = element_blank(),
      axis.title.y.left = element_text(
        size = 22,
        color = "black",
        margin = margin(
          r = 12
        )
      ),
      axis.title.y.right = element_text(
        size = 22,
        color = "black",
        margin = margin(
          l = 12
        )
      ),
      axis.text.x = element_text(
        size = 17,
        color = "black",
        lineheight = 0.95,
        margin = margin(
          t = 10
        )
      ),
      axis.text.y.left = element_text(
        size = 18,
        color = "black",
        margin = margin(
          r = 8
        )
      ),
      axis.text.y.right = element_text(
        size = 18,
        color = "black",
        margin = margin(
          l = 8
        )
      ),
      legend.position = "bottom",
      legend.box.spacing = grid::unit(
        0.50,
        "cm"
      ),
      legend.margin = margin(
        t = 0,
        r = 0,
        b = 0,
        l = 0
      ),
      legend.text = element_text(
        size = 14
      ),
      legend.key.width = grid::unit(
        1.15,
        "cm"
      ),
      legend.key.height = grid::unit(
        0.55,
        "cm"
      ),
      legend.spacing.x = grid::unit(
        0.25,
        "cm"
      ),
      plot.margin = margin(
        t = 42,
        r = 26,
        b = 16,
        l = 22
      )
    )
}


validate_configuration()

for (aggregation_method in AGGREGATION_METHODS) {
  output_subdirectory <- tolower(
    aggregation_method
  )

  output_root <- file.path(
    BASE_OUTPUT_ROOT,
    output_subdirectory
  )

  output_pdf <- file.path(
    output_root,
    paste0(
      "RQ_ablation_all_metrics_combined_dual_axis_",
      SUMMARY_STATISTIC,
      ".pdf"
    )
  )

  output_summary <- file.path(
    output_root,
    paste0(
      "RQ_ablation_all_metrics_combined_dual_axis_",
      SUMMARY_STATISTIC,
      "_summary.csv"
    )
  )

  dir.create(
    output_root,
    recursive = TRUE,
    showWarnings = FALSE
  )

  ablation_data <- load_ablation_data(
    aggregation_method
  )

  summary_data <- summarize_all_releases(
    ablation_data
  )

  prepared_data <- prepare_plot_data(
    summary_data
  )

  plot_object <- build_combined_barplot(
    plot_data = prepared_data$plot_data,
    ifa_limits = prepared_data$ifa_limits
  )

  ggsave(
    filename = output_pdf,
    plot = plot_object,
    width = OUTPUT_WIDTH,
    height = OUTPUT_HEIGHT,
    units = "in",
    dpi = OUTPUT_DPI,
    device = grDevices::cairo_pdf,
    family = "Arial",
    limitsize = FALSE
  )

  write.csv(
    prepared_data$plot_data,
    output_summary,
    row.names = FALSE,
    fileEncoding = "UTF-8"
  )
}

message(
  paste0(
    "Saved causal-token ablation figures to ",
    BASE_OUTPUT_ROOT
  )
)

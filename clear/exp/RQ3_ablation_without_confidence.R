suppressPackageStartupMessages({
  library(ggplot2)
  library(scales)
})

DATA_ROOT <- file.path(
  "..",
  "Data"
)

DATASET_NAME <- "linedp_dataset"
MODEL_NAME <- "logistic_regression"

WITH_CONFIDENCE_ROOT <- file.path(
  DATA_ROOT,
  DATASET_NAME,
  "results",
  "ranking_evaluation",
  MODEL_NAME
)

WITHOUT_CONFIDENCE_ROOT <- file.path(
  DATA_ROOT,
  DATASET_NAME,
  "results",
  "ranking_evaluation_without_confidence",
  MODEL_NAME
)

OUTPUT_ROOT <- file.path(
  ".",
  "result_fig",
  "RQ_ablation_confidence",
  DATASET_NAME
)

AGGREGATION_METHODS <- c(
  "SUM",
  "MAX",
  "TOP2AVG",
  "TOP3AVG"
)

SUMMARY_STATISTIC <- "mean"
# SUMMARY_STATISTIC <- "median"

METHOD_ORDER <- c(
  "with_confidence",
  "without_confidence"
)

METHOD_LABELS <- c(
  "with_confidence" =
    "CLEAR",
  "without_confidence" =
    "CLEAR w/o Confidence"
)

METHOD_COLORS <- c(
  "with_confidence" =
    "#DA936A",
  "without_confidence" =
    "#547DA3"
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
  "recall_20" =
    "Recall@20% (\u2191)",
  "effort@20%recall" =
    "Effort@20% (\u2193)",
  "far" =
    "FAR (\u2193)",
  "auc" =
    "AUC (\u2191)",
  "d2h" =
    "D2H (\u2193)",
  "ifa" =
    "IFA (\u2193)"
)

LEFT_Y_LIMITS <- c(
  0,
  0.8
)

IFA_RIGHT_LIMITS <- NULL
IFA_RIGHT_PADDING_RATIO <- 0.10

LEFT_AXIS_ACCURACY <- 0.01
BAR_LABEL_ACCURACY <- 0.00001
IFA_BAR_LABEL_ACCURACY <- 1

BAR_WIDTH <- 0.68
DODGE_WIDTH <- 0.78

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
  return(
    paste0(
      "CLEAR-",
      aggregation_method,
      ".csv"
    )
  )
}


get_output_path <- function(
    aggregation_method
)
{
  return(
    file.path(
      OUTPUT_ROOT,
      paste0(
        "CLEAR-",
        aggregation_method,
        ".pdf"
      )
    )
  )
}


load_metric_file <- function(
    input_path,
    method_name
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
    names(
      input_data
    )
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

  if (
    any(
      duplicated(
        input_data$release
      )
    )
  ) {
    stop(
      paste0(
        "Duplicate release rows were found in ",
        input_path,
        "."
      )
    )
  }

  for (metric_name in METRIC_ORDER) {
    input_data[[
      metric_name
    ]] <- suppressWarnings(
      as.numeric(
        input_data[[
          metric_name
        ]]
      )
    )
  }

  input_data$method <- method_name

  return(
    input_data
  )
}


load_paired_data <- function(
    aggregation_method
)
{
  input_filename <- get_input_filename(
    aggregation_method
  )

  with_confidence_file <- file.path(
    WITH_CONFIDENCE_ROOT,
    input_filename
  )

  without_confidence_file <- file.path(
    WITHOUT_CONFIDENCE_ROOT,
    input_filename
  )

  with_confidence_data <- load_metric_file(
    with_confidence_file,
    "with_confidence"
  )

  without_confidence_data <- load_metric_file(
    without_confidence_file,
    "without_confidence"
  )

  common_releases <- intersect(
    with_confidence_data$release,
    without_confidence_data$release
  )

  if (length(common_releases) == 0) {
    stop(
      paste0(
        "No common releases were found for ",
        aggregation_method,
        "."
      )
    )
  }

  missing_with_confidence <- setdiff(
    without_confidence_data$release,
    with_confidence_data$release
  )

  missing_without_confidence <- setdiff(
    with_confidence_data$release,
    without_confidence_data$release
  )

  if (length(missing_with_confidence) > 0) {
    warning(
      paste0(
        "[",
        aggregation_method,
        "] Releases missing from confidence-enhanced results: ",
        paste(
          missing_with_confidence,
          collapse = ", "
        ),
        ". These releases will be excluded."
      )
    )
  }

  if (length(missing_without_confidence) > 0) {
    warning(
      paste0(
        "[",
        aggregation_method,
        "] Releases missing from no-confidence results: ",
        paste(
          missing_without_confidence,
          collapse = ", "
        ),
        ". These releases will be excluded."
      )
    )
  }

  with_confidence_data <- with_confidence_data[
    with_confidence_data$release %in%
      common_releases,
    ,
    drop = FALSE
  ]

  without_confidence_data <- without_confidence_data[
    without_confidence_data$release %in%
      common_releases,
    ,
    drop = FALSE
  ]

  combined_data <- rbind(
    with_confidence_data,
    without_confidence_data
  )

  combined_data$method <- factor(
    combined_data$method,
    levels = METHOD_ORDER
  )

  return(
    combined_data
  )
}


get_selected_statistic <- function(
    values
)
{
  if (SUMMARY_STATISTIC == "mean") {
    return(
      mean(
        values
      )
    )
  }

  return(
    median(
      values
    )
  )
}


summarize_metrics <- function(
    input_data
)
{
  summary_rows <- list()
  row_index <- 1

  for (metric_name in METRIC_ORDER) {
    for (method_name in METHOD_ORDER) {
      metric_values <- suppressWarnings(
        as.numeric(
          input_data[
            input_data$method ==
              method_name,
            metric_name
          ]
        )
      )

      metric_values <- metric_values[
        is.finite(
          metric_values
        )
      ]

      if (length(metric_values) == 0) {
        stop(
          paste0(
            "No finite values were found for metric ",
            metric_name,
            " and method ",
            method_name,
            "."
          )
        )
      }

      summary_rows[[
        row_index
      ]] <- data.frame(
        metric = metric_name,
        method = method_name,
        selected_value =
          get_selected_statistic(
            metric_values
          ),
        mean_value = mean(
          metric_values
        ),
        median_value = median(
          metric_values
        ),
        standard_deviation = if (
          length(
            metric_values
          ) > 1
        ) {
          sd(
            metric_values
          )
        } else {
          0
        },
        release_count = length(
          metric_values
        ),
        summary_statistic =
          SUMMARY_STATISTIC,
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

  summary_data$method <- factor(
    summary_data$method,
    levels = METHOD_ORDER
  )

  summary_data <- summary_data[
    order(
      summary_data$metric,
      summary_data$method
    ),
    ,
    drop = FALSE
  ]

  return(
    summary_data
  )
}


calculate_ifa_limits <- function(
    summary_data
)
{
  if (!is.null(IFA_RIGHT_LIMITS)) {
    if (
      length(
        IFA_RIGHT_LIMITS
      ) != 2 ||
        any(
          !is.finite(
            IFA_RIGHT_LIMITS
          )
        ) ||
        any(
          IFA_RIGHT_LIMITS <= 0
        ) ||
        IFA_RIGHT_LIMITS[1] >=
          IFA_RIGHT_LIMITS[2]
    ) {
      stop(
        "IFA_RIGHT_LIMITS must contain two increasing positive values."
      )
    }

    return(
      IFA_RIGHT_LIMITS
    )
  }

  ifa_values <- summary_data$selected_value[
    as.character(
      summary_data$metric
    ) == "ifa"
  ]

  ifa_values <- ifa_values[
    is.finite(
      ifa_values
    ) &
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
      ) *
        (
          1 +
            IFA_RIGHT_PADDING_RATIO
        )
    )
  )

  if (minimum_exponent >= maximum_exponent) {
    maximum_exponent <- (
      minimum_exponent +
        1
    )
  }

  return(
    c(
      10 ^ minimum_exponent,
      10 ^ maximum_exponent
    )
  )
}


transform_ifa_to_left_axis <- function(
    values,
    ifa_limits
)
{
  if (
    any(
      !is.finite(
        values
      )
    ) ||
      any(
        values <= 0
      )
  ) {
    stop(
      paste0(
        "IFA ",
        SUMMARY_STATISTIC,
        " values must be finite and positive."
      )
    )
  }

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
    !is.finite(
      ifa_range
    ) ||
      ifa_range <= 0
  ) {
    stop(
      "IFA right-axis limits must define a positive log-scale range."
    )
  }

  transformed_values <- (
    log10(
      values
    ) -
      log_ifa_limits[1]
  ) /
    ifa_range

  return(
    transformed_values *
      left_range +
      LEFT_Y_LIMITS[1]
  )
}


prepare_plot_data <- function(
    summary_data
)
{
  ifa_limits <- calculate_ifa_limits(
    summary_data
  )

  plot_data <- summary_data
  plot_data$plot_value <- (
    plot_data$selected_value
  )

  ifa_mask <- (
    as.character(
      plot_data$metric
    ) == "ifa"
  )

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
    accuracy =
      IFA_BAR_LABEL_ACCURACY,
    big.mark = ","
  )(
    plot_data$selected_value[
      ifa_mask
    ]
  )

  return(
    list(
      plot_data = plot_data,
      ifa_limits = ifa_limits
    )
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
      fill = method
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
      size = 3.8,
      color = "black",
      na.rm = TRUE
    ) +
    scale_fill_manual(
      values = METHOD_COLORS,
      breaks = METHOD_ORDER,
      labels = METHOD_LABELS[
        METHOD_ORDER
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
        by = 0.20
      ),
      labels = label_number(
        accuracy =
          LEFT_AXIS_ACCURACY
      ),
      sec.axis = sec_axis(
        trans = ~ 10 ^ (
          (
            (
              . -
                LEFT_Y_LIMITS[1]
            ) /
              left_range
          ) *
            ifa_range +
            log_ifa_limits[1]
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
          0.03
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
        size = 15,
        color = "black",
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
        0.25,
        "cm"
      ),
      legend.margin = margin(
        t = 4,
        r = 0,
        b = 0,
        l = 0
      ),
      legend.text = element_text(
        size = 15
      ),
      legend.key.width = grid::unit(
        1.20,
        "cm"
      ),
      legend.key.height = grid::unit(
        0.55,
        "cm"
      ),
      legend.spacing.x = grid::unit(
        0.30,
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


generate_plot_for_aggregation <- function(
    aggregation_method
)
{
  paired_input_data <- load_paired_data(
    aggregation_method
  )

  summary_data <- summarize_metrics(
    paired_input_data
  )

  prepared_plot_data <- prepare_plot_data(
    summary_data
  )

  plot_object <- build_combined_barplot(
    plot_data =
      prepared_plot_data$plot_data,
    ifa_limits =
      prepared_plot_data$ifa_limits
  )

  output_path <- get_output_path(
    aggregation_method
  )

  ggsave(
    filename = output_path,
    plot = plot_object,
    width = OUTPUT_WIDTH,
    height = OUTPUT_HEIGHT,
    units = "in",
    dpi = OUTPUT_DPI,
    device = grDevices::cairo_pdf,
    family = "Arial",
    limitsize = FALSE
  )

  message(
    paste0(
      "[",
      aggregation_method,
      "] Saved confidence-ablation figure to ",
      output_path
    )
  )
}


validate_configuration()

dir.create(
  OUTPUT_ROOT,
  recursive = TRUE,
  showWarnings = FALSE
)

for (
  aggregation_method in
  AGGREGATION_METHODS
) {
  generate_plot_for_aggregation(
    aggregation_method
  )
}

message(
  paste0(
    "Saved all four confidence-ablation figures to ",
    OUTPUT_ROOT
  )
)

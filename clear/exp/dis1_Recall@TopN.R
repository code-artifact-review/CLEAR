suppressPackageStartupMessages({
  library(ggplot2)
  library(scales)
})

METRICS_ROOT <- "./metrics"

CLEAR_METRICS_ROOT <- file.path(
  "..",
  "Data"
)

MODEL_NAME <- "logistic_regression"

RESULT_ROOT <- "./result_fig/RQ_for_recall_TopN"

DATASETS <- c(
  # "glance_dataset",
  "linedp_dataset"
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

RECALL_MIN_PERCENTAGE <- 10
RECALL_MAX_PERCENTAGE <- 80
RECALL_PERCENTAGE_STEP <- 10

LEGEND_BOX_SPACING_CM <- 0
LEGEND_TOP_MARGIN <- 5

RECALL_Y_MIN <- 0.10
RECALL_Y_MAX <- 0.95
RECALL_Y_STEP <- 0.1

RECALL_PERCENTAGES <- seq(
  RECALL_MIN_PERCENTAGE,
  RECALL_MAX_PERCENTAGE,
  by = RECALL_PERCENTAGE_STEP
)

RECALL_LINE_COLORS <- c(
  "CLEAR-MAX" = "#B85C38",
  "CLEAR-SUM" = "#D47A52",
  "CLEAR-TOP2AVG" = "#E49B73",
  "CLEAR-TOP3AVG" = "#F0B894",
  "SOUND-Barinel" = "#244A6E",
  "SOUND-Dstar" = "#315D84",
  "SOUND-Ochiai" = "#3E709A",
  "SOUND-Op2" = "#4C83AF",
  "SOUND-Tarantula" = "#5A96C4",
  "DeepLineDP" = "#729FC4",
  "LineDP" = "#829FC4",
  "GLANCE-LR" = "#86ACCB",
  "GLANCE-EA" = "#9AB9D2",
  "GLANCE-MD" = "#AEC6D9",
  "ErrorProne" = "#C2D3E0",
  "Ngram" = "#D6E0E7"
)

RECALL_LINE_SHAPES <- c(
  "CLEAR-MAX" = 16,
  "CLEAR-SUM" = 17,
  "CLEAR-TOP2AVG" = 15,
  "CLEAR-TOP3AVG" = 18,
  "SOUND-Barinel" = 0,
  "SOUND-Dstar" = 1,
  "SOUND-Ochiai" = 2,
  "SOUND-Op2" = 3,
  "SOUND-Tarantula" = 4,
  "DeepLineDP" = 5,
  "GLANCE-LR" = 6,
  "GLANCE-EA" = 7,
  "GLANCE-MD" = 8,
  "ErrorProne" = 9,
  "Ngram" = 10,
  "LineDP" = 11
)


load_recall_curve_data <- function(
    dataset_directory,
    dataset_name
)
{
  recall_columns <- paste0(
    "recall_",
    RECALL_PERCENTAGES
  )
  
  recall_curve_data <- data.frame(
    percentage = numeric(),
    mean_recall = numeric(),
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
    
    missing_columns <- setdiff(
      recall_columns,
      names(current_data)
    )
    
    if (length(missing_columns) > 0) {
      next
    }
    
    recall_means <- vapply(
      recall_columns,
      function(column_name) {
        values <- suppressWarnings(
          as.numeric(
            current_data[[column_name]]
          )
        )
        
        if (!any(is.finite(values))) {
          return(NA_real_)
        }
        
        mean(
          values,
          na.rm = TRUE
        )
      },
      numeric(1)
    )
    
    current_curve_data <- data.frame(
      percentage = RECALL_PERCENTAGES,
      mean_recall = recall_means,
      group = sub(
        "\\.csv$",
        "",
        file_name
      ),
      stringsAsFactors = FALSE
    )
    
    current_curve_data <- current_curve_data[
      is.finite(current_curve_data$mean_recall),
      ,
      drop = FALSE
    ]
    
    recall_curve_data <- rbind(
      recall_curve_data,
      current_curve_data
    )
  }
  
  if (nrow(recall_curve_data) == 0) {
    stop(
      paste0(
        "No valid recall data were found from ",
        RECALL_MIN_PERCENTAGE,
        "% to ",
        RECALL_MAX_PERCENTAGE,
        "% in ",
        dataset_directory,
        "."
      )
    )
  }
  
  available_methods <- METHOD_ORDER[
    METHOD_ORDER %in% unique(
      recall_curve_data$group
    )
  ]
  
  recall_curve_data$group <- factor(
    recall_curve_data$group,
    levels = available_methods
  )
  
  return(recall_curve_data)
}


build_recall_curve_plot <- function(recall_curve_data)
{
  available_methods <- levels(
    recall_curve_data$group
  )
  
  line_colors <- RECALL_LINE_COLORS[
    available_methods
  ]
  line_shapes <- RECALL_LINE_SHAPES[
    available_methods
  ]
  
  missing_colors <- names(line_colors)[
    is.na(line_colors)
  ]
  missing_shapes <- names(line_shapes)[
    is.na(line_shapes)
  ]
  
  if (length(missing_colors) > 0) {
    fallback_colors <- grDevices::hcl.colors(
      length(missing_colors),
      palette = "Dark 3"
    )
    line_colors[
      missing_colors
    ] <- fallback_colors
  }
  
  if (length(missing_shapes) > 0) {
    line_shapes[
      missing_shapes
    ] <- seq_along(
      missing_shapes
    ) %% 20
  }
  
  ggplot(
    recall_curve_data,
    aes(
      x = percentage,
      y = mean_recall,
      color = group,
      shape = group,
      group = group
    )
  ) +
    geom_line(
      linewidth = 1.15,
      alpha = 0.95,
      na.rm = TRUE
    ) +
    geom_point(
      size = 2.6,
      stroke = 0.45,
      na.rm = TRUE
    ) +
    scale_color_manual(
      values = line_colors,
      breaks = available_methods,
      drop = FALSE
    ) +
    scale_shape_manual(
      values = line_shapes,
      breaks = available_methods,
      drop = FALSE
    ) +
    scale_x_continuous(
      breaks = RECALL_PERCENTAGES,
      limits = c(
        RECALL_MIN_PERCENTAGE,
        RECALL_MAX_PERCENTAGE
      ),
      labels = function(values) {
        paste0(
          values,
          "%"
        )
      },
      expand = expansion(
        mult = c(
          0.01,
          0.01
        )
      )
    ) +
    scale_y_continuous(
      breaks = seq(
        RECALL_Y_MIN,
        RECALL_Y_MAX,
        by = RECALL_Y_STEP
      ),
      labels = label_number(
        accuracy = 0.01
      ),
      expand = expansion(
        mult = c(
          0,
          0.02
        )
      )
    ) +
    coord_cartesian(
      ylim = c(
        RECALL_Y_MIN,
        RECALL_Y_MAX
      )
    ) +
    guides(
      color = guide_legend(
        title = NULL,
        nrow = 3,
        byrow = TRUE,
        override.aes = list(
          linewidth = 1.4,
          size = 3
        )
      ),
      shape = guide_legend(
        title = NULL,
        nrow = 3,
        byrow = TRUE
      )
    ) +
    labs(
      x = NULL,
      y = "Recall@TopN%LOC",
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
      axis.title.x = element_blank(),
      axis.title.y = element_text(
        size = 22,
        color = "black",
        margin = margin(
          r = 12
        )
      ),
      axis.text.x = element_text(
        size = 18,
        color = "black",
        margin = margin(
          t = 10
        )
      ),
      axis.text.y = element_text(
        size = 18,
        color = "black",
        margin = margin(
          r = 10
        )
      ),
      legend.position = "bottom",
      legend.box.spacing = grid::unit(
        LEGEND_BOX_SPACING_CM,
        "cm"
      ),
      legend.margin = margin(
        t = LEGEND_TOP_MARGIN,
        r = 0,
        b = 0,
        l = 0
      ),
      legend.text = element_text(
        size = 14
      ),
      legend.key.width = grid::unit(
        1.5,
        "cm"
      ),
      legend.spacing.x = grid::unit(
        0.25,
        "cm"
      ),
      plot.margin = margin(
        t = 12,
        r = 18,
        b = 12,
        l = 18
      )
    )
}


save_recall_curve_plot <- function(
    plot_object,
    output_path
)
{
  ggsave(
    filename = output_path,
    plot = plot_object,
    width = 16,
    height = 9,
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
  
  recall_curve_data <- load_recall_curve_data(
    dataset_directory,
    dataset_name
  )
  
  available_methods <- levels(
    recall_curve_data$group
  )
  
  recall_curve_csv_data <- data.frame(
    Method = available_methods,
    stringsAsFactors = FALSE,
    check.names = FALSE
  )
  
  for (percentage in RECALL_PERCENTAGES) {
    recall_curve_csv_data[[paste0(percentage, "%")]] <- vapply(
      available_methods,
      function(method_name) {
        values <- recall_curve_data$mean_recall[
          recall_curve_data$group == method_name &
            recall_curve_data$percentage == percentage
        ]
        
        if (length(values) == 0) {
          return(NA_real_)
        }
        
        values[1]
      },
      numeric(1)
    )
  }
  
  recall_curve_csv_output_path <- file.path(
    dataset_result_directory,
    "RQ_recall_curve.csv"
  )
  
  write.csv(
    recall_curve_csv_data,
    recall_curve_csv_output_path,
    row.names = FALSE
  )
  
  recall_curve_plot <- build_recall_curve_plot(
    recall_curve_data
  )
  
  recall_curve_output_path <- file.path(
    dataset_result_directory,
    "RQ_recall_curve.pdf"
  )
  
  save_recall_curve_plot(
    plot_object = recall_curve_plot,
    output_path = recall_curve_output_path
  )
}

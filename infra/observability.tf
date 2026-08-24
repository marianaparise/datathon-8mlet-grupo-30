# Observabilidade.
#
# Os alarmes vigiam três coisas diferentes: se o serviço está de pé, se está
# respondendo, e — a que importa para um bandit — se a conversão desabou. As
# duas primeiras qualquer API tem. A terceira é específica: uma política pode
# estar perfeitamente saudável em CPU e latência enquanto recomenda o braço
# errado para todo mundo.

resource "aws_cloudwatch_log_group" "api" {
  name              = "/ecs/${local.name}"
  retention_in_days = var.log_retention_days
}

resource "aws_cloudwatch_log_group" "firehose" {
  name              = "/aws/kinesisfirehose/${local.name}"
  retention_in_days = var.log_retention_days
}

resource "aws_sns_topic" "alarms" {
  count = var.alarm_email == "" ? 0 : 1
  name  = "${local.name}-alarms"
}

resource "aws_sns_topic_subscription" "alarms_email" {
  count = var.alarm_email == "" ? 0 : 1

  topic_arn = aws_sns_topic.alarms[0].arn
  protocol  = "email"
  endpoint  = var.alarm_email
}

locals {
  alarm_actions = var.alarm_email == "" ? [] : [aws_sns_topic.alarms[0].arn]
}

resource "aws_cloudwatch_metric_alarm" "unhealthy_hosts" {
  alarm_name          = "${local.name}-alvos-fora"
  alarm_description   = "Alguma task deixou de passar no health check do ALB."
  namespace           = "AWS/ApplicationELB"
  metric_name         = "UnHealthyHostCount"
  statistic           = "Maximum"
  period              = 60
  evaluation_periods  = 2
  threshold           = 0
  comparison_operator = "GreaterThanThreshold"
  treat_missing_data  = "notBreaching"

  dimensions = {
    LoadBalancer = aws_lb.main.arn_suffix
    TargetGroup  = aws_lb_target_group.api.arn_suffix
  }

  alarm_actions = local.alarm_actions
  ok_actions    = local.alarm_actions
}

resource "aws_cloudwatch_metric_alarm" "server_errors" {
  alarm_name          = "${local.name}-erros-5xx"
  alarm_description   = "A aplicação passou a devolver erro."
  namespace           = "AWS/ApplicationELB"
  metric_name         = "HTTPCode_Target_5XX_Count"
  statistic           = "Sum"
  period              = 300
  evaluation_periods  = 1
  threshold           = 10
  comparison_operator = "GreaterThanThreshold"
  treat_missing_data  = "notBreaching"

  dimensions = {
    LoadBalancer = aws_lb.main.arn_suffix
  }

  alarm_actions = local.alarm_actions
}

resource "aws_cloudwatch_metric_alarm" "latency" {
  alarm_name          = "${local.name}-latencia-p99"
  alarm_description   = "p99 acima de 1s — pontuar seis braços deve levar milissegundos."
  namespace           = "AWS/ApplicationELB"
  metric_name         = "TargetResponseTime"
  extended_statistic  = "p99"
  period              = 300
  evaluation_periods  = 2
  threshold           = 1
  comparison_operator = "GreaterThanThreshold"
  treat_missing_data  = "notBreaching"

  dimensions = {
    LoadBalancer = aws_lb.main.arn_suffix
  }

  alarm_actions = local.alarm_actions
}

# O alarme que existe por ser um bandit, e não uma API qualquer.
#
# Depende de a aplicação publicar a métrica `ConversionRate` — o que só é
# possível depois que o ciclo de feedback existir. Enquanto isso ele fica em
# INSUFFICIENT_DATA, e `treat_missing_data = "missing"` impede que isso vire
# alarme falso.
resource "aws_cloudwatch_metric_alarm" "conversion_drop" {
  alarm_name          = "${local.name}-queda-de-conversao"
  alarm_description   = <<-EOT
    Conversão observada abaixo da taxa-base histórica de 11,27%.

    É o sinal de que a política degradou — por deriva de distribuição, por
    artefato errado no deploy, ou porque o mundo mudou. Nenhuma métrica de
    infraestrutura pega isso: CPU e latência ficam perfeitas enquanto o modelo
    recomenda o braço errado para todo mundo.
  EOT
  namespace           = var.project_name
  metric_name         = "ConversionRate"
  statistic           = "Average"
  period              = 3600
  evaluation_periods  = 3
  threshold           = 0.1127
  comparison_operator = "LessThanThreshold"
  treat_missing_data  = "missing"

  alarm_actions = local.alarm_actions
}

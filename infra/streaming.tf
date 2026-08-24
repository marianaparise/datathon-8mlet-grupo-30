# O caminho de volta: decisão tomada → desfecho observado → S3.
#
# É a peça que fecha o ciclo do bandit. Sem ela a política decide e nunca fica
# sabendo o que aconteceu, que é exatamente a limitação declarada no README:
# o `?explore=true` da API explora, mas não aprende.
#
# Firehose em vez de escrita direta no S3 porque conversão chega em eventos
# pequenos e frequentes; o buffer agrega antes de gravar, evitando milhões de
# objetos minúsculos que tornariam a releitura cara.

resource "aws_kinesis_firehose_delivery_stream" "rewards" {
  name        = "${local.name}-rewards"
  destination = "extended_s3"

  extended_s3_configuration {
    role_arn   = aws_iam_role.firehose.arn
    bucket_arn = aws_s3_bucket.rewards.arn

    # Particionar por dia é o que permite reprocessar uma janela sem varrer o
    # bucket inteiro.
    prefix              = "rewards/dt=!{timestamp:yyyy-MM-dd}/"
    error_output_prefix = "errors/!{firehose:error-output-type}/dt=!{timestamp:yyyy-MM-dd}/"

    # 5 MiB ou 5 minutos, o que vier primeiro. Latência de minutos é irrelevante
    # aqui: a reavaliação de política é diária, não em tempo real.
    buffering_size     = 5
    buffering_interval = 300
    compression_format = "GZIP"

    cloudwatch_logging_options {
      enabled         = true
      log_group_name  = aws_cloudwatch_log_group.firehose.name
      log_stream_name = "delivery"
    }
  }
}

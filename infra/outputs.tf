output "api_url" {
  description = "Endereço público do serviço. /docs abre o Swagger."
  value       = "http://${aws_lb.main.dns_name}"
}

output "ecr_repository_url" {
  description = "Destino do push da imagem."
  value       = aws_ecr_repository.api.repository_url
}

output "artifacts_bucket" {
  description = "Bucket de dataset tratado, modelos e runs do MLflow."
  value       = aws_s3_bucket.artifacts.id
}

output "rewards_bucket" {
  description = "Bucket onde o Firehose entrega o log de recompensas."
  value       = aws_s3_bucket.rewards.id
}

output "arm_state_table" {
  description = "Tabela DynamoDB com as posteriores por braço."
  value       = aws_dynamodb_table.arm_state.name
}

output "reward_stream" {
  description = "Stream do Firehose para publicar desfechos observados."
  value       = aws_kinesis_firehose_delivery_stream.rewards.name
}

output "log_group" {
  description = "Grupo de logs da aplicação no CloudWatch."
  value       = aws_cloudwatch_log_group.api.name
}

output "push_image_commands" {
  description = "Sequência para publicar a imagem local no ECR."
  value       = <<-EOT
    aws ecr get-login-password --region ${var.aws_region} \
      | docker login --username AWS --password-stdin ${aws_ecr_repository.api.repository_url}
    docker build -t ${aws_ecr_repository.api.repository_url}:${var.image_tag} .
    docker push ${aws_ecr_repository.api.repository_url}:${var.image_tag}
    aws ecs update-service --cluster ${aws_ecs_cluster.main.name} \
      --service ${aws_ecs_service.api.name} --force-new-deployment \
      --region ${var.aws_region}
  EOT
}

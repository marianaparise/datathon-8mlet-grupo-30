variable "aws_region" {
  description = "Região onde tudo é provisionado."
  type        = string
  default     = "us-east-1"
}

variable "project_name" {
  description = "Prefixo de nome de todo recurso. Precisa ser válido em DNS."
  type        = string
  default     = "tc5-bandit"

  validation {
    condition     = can(regex("^[a-z][a-z0-9-]{2,24}$", var.project_name))
    error_message = "Use minúsculas, números e hífen, começando por letra (3 a 25 caracteres)."
  }
}

variable "environment" {
  description = "Ambiente lógico, usado em tags e nomes."
  type        = string
  default     = "dev"

  validation {
    condition     = contains(["dev", "staging", "prod"], var.environment)
    error_message = "Ambientes aceitos: dev, staging, prod."
  }
}

variable "vpc_cidr" {
  description = "Bloco CIDR da VPC."
  type        = string
  default     = "10.20.0.0/16"
}

variable "image_tag" {
  description = "Tag da imagem no ECR que o serviço deve rodar."
  type        = string
  default     = "latest"
}

variable "task_cpu" {
  description = "Unidades de CPU da task Fargate. 512 = 0,5 vCPU."
  type        = number
  default     = 512
}

variable "task_memory" {
  description = "Memória em MiB. scikit-learn e o artefato pedem folga."
  type        = number
  default     = 1024
}

variable "desired_count" {
  description = "Réplicas do serviço. Duas cobrem a queda de uma AZ."
  type        = number
  default     = 2

  validation {
    condition     = var.desired_count >= 1
    error_message = "É preciso ao menos uma réplica."
  }
}

variable "log_retention_days" {
  description = "Retenção dos logs da aplicação no CloudWatch."
  type        = number
  default     = 30
}

variable "reward_retention_days" {
  description = <<-EOT
    Retenção do log de recompensas no S3, em dias.

    Prazo curto de propósito: o log guarda decisão e desfecho por cliente, e a
    seção de governança do README define 180 dias como o necessário para
    reavaliar uma política. Guardar além disso é acúmulo sem finalidade.
  EOT
  type        = number
  default     = 180
}

variable "alarm_email" {
  description = "Destino das notificações de alarme. Vazio desliga o SNS."
  type        = string
  default     = ""
}

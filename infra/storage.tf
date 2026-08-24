# Armazenamento: artefatos de modelo, log de recompensas e estado dos braços.

resource "aws_ecr_repository" "api" {
  name                 = local.name
  image_tag_mutability = "MUTABLE"

  image_scanning_configuration {
    scan_on_push = true
  }

  encryption_configuration {
    encryption_type = "AES256"
  }
}

# Sem isto o repositório acumula toda imagem já publicada, indefinidamente.
resource "aws_ecr_lifecycle_policy" "api" {
  repository = aws_ecr_repository.api.name

  policy = jsonencode({
    rules = [{
      rulePriority = 1
      description  = "Mantem as 10 imagens mais recentes"
      selection = {
        tagStatus   = "any"
        countType   = "imageCountMoreThan"
        countNumber = 10
      }
      action = { type = "expire" }
    }]
  })
}

# --- Artefatos: dataset tratado, ambiente calibrado, runs do MLflow ----------

resource "aws_s3_bucket" "artifacts" {
  bucket = local.artifacts_bucket
}

# Versionamento é o que permite voltar para um modelo anterior sem retreinar.
resource "aws_s3_bucket_versioning" "artifacts" {
  bucket = aws_s3_bucket.artifacts.id

  versioning_configuration {
    status = "Enabled"
  }
}

resource "aws_s3_bucket_server_side_encryption_configuration" "artifacts" {
  bucket = aws_s3_bucket.artifacts.id

  rule {
    apply_server_side_encryption_by_default {
      sse_algorithm = "AES256"
    }
  }
}

resource "aws_s3_bucket_public_access_block" "artifacts" {
  bucket = aws_s3_bucket.artifacts.id

  block_public_acls       = true
  block_public_policy     = true
  ignore_public_acls      = true
  restrict_public_buckets = true
}

# --- Log de recompensas: a decisão tomada e o desfecho observado -------------

resource "aws_s3_bucket" "rewards" {
  bucket = local.rewards_bucket
}

resource "aws_s3_bucket_server_side_encryption_configuration" "rewards" {
  bucket = aws_s3_bucket.rewards.id

  rule {
    apply_server_side_encryption_by_default {
      sse_algorithm = "AES256"
    }
  }
}

resource "aws_s3_bucket_public_access_block" "rewards" {
  bucket = aws_s3_bucket.rewards.id

  block_public_acls       = true
  block_public_policy     = true
  ignore_public_acls      = true
  restrict_public_buckets = true
}

# Retenção é requisito de governança, não faxina: o log guarda decisão e
# desfecho por cliente. Ver a seção de governança do README.
resource "aws_s3_bucket_lifecycle_configuration" "rewards" {
  bucket = aws_s3_bucket.rewards.id

  rule {
    id     = "retencao-do-log-de-recompensas"
    status = "Enabled"

    filter {}

    expiration {
      days = var.reward_retention_days
    }
  }
}

# --- Estado dos braços ------------------------------------------------------

# É isto que falta para o `?explore=true` da API deixar de ser sem estado: as
# posteriores Beta por braço vivem aqui, e cada conversão observada atualiza a
# linha correspondente. Sem esta tabela o bandit reinicia a cada deploy.
resource "aws_dynamodb_table" "arm_state" {
  name         = "${local.name}-arm-state"
  billing_mode = "PAY_PER_REQUEST"
  hash_key     = "arm"

  attribute {
    name = "arm"
    type = "S"
  }

  point_in_time_recovery {
    enabled = true
  }

  server_side_encryption {
    enabled = true
  }
}

terraform {
  required_version = ">= 1.6"

  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "~> 5.60"
    }
  }

  # Backend remoto fica comentado de propósito: um bucket de state precisa
  # existir antes do primeiro apply, e criá-lo com o mesmo Terraform que o usa
  # é um problema de ovo e galinha. Para a avaliação, state local basta.
  #
  # backend "s3" {
  #   bucket         = "tc5-terraform-state"
  #   key            = "tc5/terraform.tfstate"
  #   region         = "us-east-1"
  #   dynamodb_table = "tc5-terraform-locks"
  #   encrypt        = true
  # }
}

provider "aws" {
  region = var.aws_region

  default_tags {
    tags = {
      Project     = var.project_name
      Environment = var.environment
      ManagedBy   = "terraform"
      Course      = "POSTECH-MLET-Datathon"
    }
  }
}

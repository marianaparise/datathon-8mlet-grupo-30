# Rede mínima e explícita.
#
# Só sub-redes públicas, sem NAT Gateway. Não é descuido: um NAT custa cerca de
# 32 USD por mês por AZ mesmo parado, e o único tráfego de saída aqui é o pull
# da imagem do ECR. As tasks ficam em sub-rede pública com IP público, e o que
# protege o serviço é o security group — que só aceita entrada vinda do ALB.
#
# Para produção de verdade a escolha se inverte: sub-redes privadas com VPC
# endpoints para ECR, S3 e CloudWatch, sem IP público em task nenhuma.

resource "aws_vpc" "main" {
  cidr_block           = var.vpc_cidr
  enable_dns_support   = true
  enable_dns_hostnames = true

  tags = { Name = local.name }
}

resource "aws_internet_gateway" "main" {
  vpc_id = aws_vpc.main.id

  tags = { Name = local.name }
}

resource "aws_subnet" "public" {
  count = length(local.azs)

  vpc_id                  = aws_vpc.main.id
  cidr_block              = cidrsubnet(var.vpc_cidr, 8, count.index)
  availability_zone       = local.azs[count.index]
  map_public_ip_on_launch = true

  tags = { Name = "${local.name}-public-${local.azs[count.index]}" }
}

resource "aws_route_table" "public" {
  vpc_id = aws_vpc.main.id

  route {
    cidr_block = "0.0.0.0/0"
    gateway_id = aws_internet_gateway.main.id
  }

  tags = { Name = "${local.name}-public" }
}

resource "aws_route_table_association" "public" {
  count = length(aws_subnet.public)

  subnet_id      = aws_subnet.public[count.index].id
  route_table_id = aws_route_table.public.id
}

resource "aws_security_group" "alb" {
  name        = "${local.name}-alb"
  description = "Entrada HTTP da internet no load balancer"
  vpc_id      = aws_vpc.main.id

  tags = { Name = "${local.name}-alb" }
}

resource "aws_vpc_security_group_ingress_rule" "alb_http" {
  security_group_id = aws_security_group.alb.id
  description       = "HTTP publico"

  cidr_ipv4   = "0.0.0.0/0"
  from_port   = 80
  to_port     = 80
  ip_protocol = "tcp"
}

resource "aws_vpc_security_group_egress_rule" "alb_all" {
  security_group_id = aws_security_group.alb.id
  description       = "Saida para as tasks"

  cidr_ipv4   = "0.0.0.0/0"
  ip_protocol = "-1"
}

resource "aws_security_group" "service" {
  name        = "${local.name}-service"
  description = "Tasks da API"
  vpc_id      = aws_vpc.main.id

  tags = { Name = "${local.name}-service" }
}

# A porta da aplicação nunca é exposta à internet: só o ALB alcança a task.
resource "aws_vpc_security_group_ingress_rule" "service_from_alb" {
  security_group_id = aws_security_group.service.id
  description       = "Somente o ALB alcanca a aplicacao"

  referenced_security_group_id = aws_security_group.alb.id
  from_port                    = local.container_port
  to_port                      = local.container_port
  ip_protocol                  = "tcp"
}

resource "aws_vpc_security_group_egress_rule" "service_all" {
  security_group_id = aws_security_group.service.id
  description       = "Pull de imagem do ECR e chamadas aos servicos AWS"

  cidr_ipv4   = "0.0.0.0/0"
  ip_protocol = "-1"
}

# ===========================================
# Makefile - Web Tech Scraper Docker Commands
# ===========================================

.PHONY: help build up down logs shell test clean rebuild

# Variables
DOCKER_COMPOSE = docker compose
CONTAINER_NAME = tech-news-scraper

help: ## Mostrar esta ayuda
	@echo "Comandos disponibles:"
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-15s\033[0m %s\n", $$1, $$2}'

build: ## Construir la imagen Docker
	$(DOCKER_COMPOSE) build

up: ## Levantar los servicios
	$(DOCKER_COMPOSE) up -d

up-logs: ## Levantar los servicios con logs visibles
	$(DOCKER_COMPOSE) up

down: ## Detener los servicios
	$(DOCKER_COMPOSE) down

restart: ## Reiniciar los servicios
	$(DOCKER_COMPOSE) restart

logs: ## Ver logs del contenedor
	$(DOCKER_COMPOSE) logs -f

shell: ## Abrir shell en el contenedor
	docker exec -it $(CONTAINER_NAME) /bin/bash

test: ## Ejecutar tests dentro del contenedor
	docker exec -it $(CONTAINER_NAME) pytest tests/ -v

clean: ## Limpiar contenedores, imágenes y volúmenes
	$(DOCKER_COMPOSE) down -v --rmi local
	docker system prune -f

rebuild: down build up ## Reconstruir y levantar servicios

status: ## Ver estado de los servicios
	$(DOCKER_COMPOSE) ps

health: ## Verificar health del servicio
	curl -s http://localhost:8000/health | python3 -m json.tool || echo "Servicio no disponible"

init: ## Inicializar proyecto (copiar .env y construir)
	@if [ ! -f .env ]; then cp .env.example .env && echo "✅ Archivo .env creado. Configura tus credenciales."; else echo "⚠️  El archivo .env ya existe."; fi
	$(DOCKER_COMPOSE) build

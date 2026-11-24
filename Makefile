.PHONY: up down

up:
	@test -f .env || cp .env.example .env
	docker compose up --build -d

down:
	docker compose down

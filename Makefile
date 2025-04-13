.PHONY: backend backend-build backend-stop backend-logs

# Start the backend service
backend:
	docker-compose up -d backend

# Build or rebuild the backend service
backend-build:
	docker-compose build backend

# Stop the backend service
backend-stop:
	docker-compose stop backend

# View backend logs
backend-logs:
	docker-compose logs -f backend

# Clean up containers
clean:
	docker-compose down
FROM node:18-alpine AS builder
WORKDIR /app
COPY package*.json ./
RUN npm ci
COPY . .
RUN npm run build

FROM nginx:alpine
# ARG con default para docker-compose local; Railway lo sobreescribe en build vars
ARG BACKEND_HOST=backend:3001
COPY --from=builder /app/dist /usr/share/nginx/html
COPY nginx.conf /etc/nginx/conf.d/nginx.template
# Sustituir en BUILD TIME — sin CMD personalizado, sin runtime env
RUN apk add --no-cache gettext \
 && envsubst '${BACKEND_HOST}' < /etc/nginx/conf.d/nginx.template \
    > /etc/nginx/conf.d/default.conf \
 && rm /etc/nginx/conf.d/nginx.template
EXPOSE 80

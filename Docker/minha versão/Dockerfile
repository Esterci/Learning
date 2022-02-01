FROM node

WORKDIR /app

COPY package*.json .

RUn npm install

COPY . .

EXPOSE 3000

CMD ["node","app.js"]
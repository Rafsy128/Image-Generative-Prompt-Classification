from locust import HttpUser, task, between

class WebsiteUser(HttpUser):
    wait_time = between(1, 5)  # Waktu tunggu antar-tugas dalam detik

    @task
    def index(self):
        self.client.get("/")  # Menguji endpoint GET
    
    @task
    def submit_data(self):
        self.client.post("/submit", json={"key": "1, 5"})  # Menguji endpoint POST

    @task
    def update_data(self):
        self.client.put("/update", json={"key": "1, 5", "new_value": "10"})  # Menguji endpoint PUT

    @task
    def delete_data(self):
        self.client.delete("/delete", json={"key": "1, 5"})  # Menguji endpoint DELETE

    @task
    def patch_data(self):
        self.client.patch("/patch", json={"key": "1, 5", "patch_value": "7"})  # Menguji endpoint PATCH

"""Frontend-only HTTP client; contains no models, data loaders or references."""
import time
from urllib.parse import quote,urlparse
import requests


class APIClient:
    def __init__(self,url,timeout=30):
        parsed=urlparse(url)
        if parsed.scheme not in ('http','https') or not parsed.netloc:raise ValueError('Use an http:// or https:// backend URL.')
        self.url=url.rstrip('/');self.timeout=timeout
        self.session=requests.Session();self.session.trust_env=False

    def request(self,path,method='GET',timeout=None):
        try:
            r=self.session.request(method,self.url+path,timeout=timeout or self.timeout)
            if not r.ok:
                try:detail=r.json().get('detail','Request failed')
                except ValueError:detail='Backend returned an invalid response'
                raise RuntimeError(f'{r.status_code}: {detail}')
            return r.json()
        except requests.Timeout as exc:raise RuntimeError('Backend request timed out. Check the job or retry.') from exc
        except requests.RequestException as exc:raise RuntimeError('Backend is unreachable. Check its LAN IP, port and connection.') from exc
        except ValueError as exc:raise RuntimeError('Backend returned invalid JSON. Verify the backend URL.') from exc

    def submit(self,trip):
        result=self.request('/predict/'+quote(trip,safe=''),method='POST')
        if not result.get('job_id'):raise RuntimeError('Backend did not return a prediction job ID.')
        return result['job_id']

    def wait(self,jid,timeout=900,interval=.5):
        start=time.monotonic()
        while time.monotonic()-start<timeout:
            status=self.request('/jobs/'+quote(jid,safe=''))
            if status['status']=='completed':return self.request('/results/'+jid)
            if status['status']=='failed':raise RuntimeError(status.get('safe_error','Prediction failed.'))
            time.sleep(interval)
        raise RuntimeError('Prediction job timed out; the backend may still be working. Retry checking its status.')

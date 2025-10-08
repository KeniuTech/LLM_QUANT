"""Value and risk factor implementations."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Sequence
import numpy as np

from app.core.indicators import normalize 
from app.utils.logging import get_logger

LOGGER = get_logger(__name__)
LOG_EXTRA = {"stage": "value_risk_factors"}


class ValueRiskFactors:
    """Value and risk factor calculation implementation.
    
    This class implements:
    1. Multi-dimensional valuation score (val_multiscore)
    2. Risk penalty factor (risk_penalty)
    """
    
    def __init__(self):
        """Initialize the calculator"""
        # Weights for different valuation metrics
        self.valuation_weights = {
            'pe': 0.3,      # PE ratio weight
            'pb': 0.3,      # PB ratio weight
            'ps': 0.2,      # PS ratio weight  
            'dv': 0.2       # Dividend yield weight
        }
    
    def compute_val_multiscore(self,
                             pe: Optional[float],
                             pb: Optional[float], 
                             ps: Optional[float],
                             dv: Optional[float]) -> Optional[float]:
        """Compute multi-dimensional valuation score
        
        Args:
            pe: PE ratio
            pb: PB ratio
            ps: PS ratio
            dv: Dividend yield
            
        Returns:
            Normalized valuation score in [-1, 1] range,
            where -1 indicates overvalued and 1 indicates undervalued
        """
        try:
            scores = []
            weights = []
            
            # PE ratio score (inverted)
            if pe is not None and pe > 0:
                pe_score = -normalize(pe, factor=25.0)  # Center around PE=25
                scores.append(pe_score)
                weights.append(self.valuation_weights['pe'])
                
            # PB ratio score (inverted)    
            if pb is not None and pb > 0:
                pb_score = -normalize(pb, factor=2.0)  # Center around PB=2
                scores.append(pb_score)
                weights.append(self.valuation_weights['pb'])
                
            # PS ratio score (inverted)
            if ps is not None and ps > 0:
                ps_score = -normalize(ps, factor=2.0)  # Center around PS=2
                scores.append(ps_score)
                weights.append(self.valuation_weights['ps'])
                
            # Dividend yield score
            if dv is not None and dv >= 0:
                dv_score = normalize(dv, factor=0.03)  # Center around 3% yield
                scores.append(dv_score)
                weights.append(self.valuation_weights['dv'])
                
            if not scores:
                return None
                
            # Normalize weights
            weights = np.array(weights)
            weights = weights / weights.sum()
            
            # Weighted average score
            return float(np.average(scores, weights=weights))
            
        except Exception as e:
            LOGGER.error(
                "Error calculating val_multiscore: %s",
                str(e),
                exc_info=True,
                extra=LOG_EXTRA
            )
            return None
            
    def compute_risk_penalty(self,
                           volatility: Optional[float],
                           turnover: Optional[float],
                           price: Optional[float],
                           avg_price: Optional[float]) -> Optional[float]:
        """Compute risk penalty factor
        
        Args:
            volatility: Historical volatility
            turnover: Turnover rate
            price: Current price
            avg_price: Moving average price (e.g. 20-day MA)
            
        Returns:
            Risk penalty score in [0, 1] range,
            where 0 indicates low risk and 1 indicates high risk
        """
        try:
            penalties = []
            
            # Volatility penalty
            if volatility is not None:
                vol_penalty = normalize(volatility, factor=0.3)  # Baseline 30% annualized vol
                penalties.append(vol_penalty)
                
            # Turnover penalty    
            if turnover is not None:
                turn_penalty = normalize(turnover, factor=5.0)  # Baseline 500% turnover
                penalties.append(turn_penalty)
                
            # Price deviation penalty
            if price is not None and avg_price is not None and avg_price > 0:
                deviation = abs(price / avg_price - 1.0)
                dev_penalty = normalize(deviation, factor=0.1)  # Baseline 10% deviation
                penalties.append(dev_penalty)
                
            if not penalties:
                return None
                
            # Average of all penalties
            return float(np.mean(penalties))
            
        except Exception as e:
            LOGGER.error(
                "Error calculating risk_penalty: %s",
                str(e),
                exc_info=True,
                extra=LOG_EXTRA
            )
            return None
            
    def compute_batch(self,
                     broker,
                     ts_codes: list[str],
                     trade_date: str,
                     batch_size: int = 100) -> None:
        """Batch compute factors for multiple stocks
        
        Args:
            broker: Data broker instance
            ts_codes: List of stock codes
            trade_date: Trading date
            batch_size: Batch size for processing
        """
        # Prepare SQL statements
        columns = ['risk_penalty', 'val_multiscore']
        insert_columns = ['ts_code', 'trade_date', 'updated_at'] + columns
        
        placeholders = ','.join('?' * len(insert_columns))
        update_clause = ', '.join(
            f"{column}=excluded.{column}"
            for column in ['updated_at'] + columns
        )
        
        sql = (
            f"INSERT INTO factors ({','.join(insert_columns)}) "
            f"VALUES ({placeholders}) "
            f"ON CONFLICT (ts_code, trade_date) DO UPDATE SET {update_clause}"
        )
        
        from datetime import datetime, timezone
        now = datetime.now(timezone.utc).isoformat()
        
        # Process in batches
        processed = 0
        for i in range(0, len(ts_codes), batch_size):
            batch = ts_codes[i:i + batch_size]
            values = []
            
            for ts_code in batch:
                try:
                    # Get required data
                    data = broker.get_stock_data(
                        ts_code,
                        trade_date,
                        fields=[
                            'daily_basic.pe',
                            'daily_basic.pb',
                            'daily_basic.ps',
                            'daily_basic.dv_ratio',
                            'factors.volat_20',
                            'daily_basic.turnover_rate',
                            'daily.close',
                            'daily_basic.pe_ttm'
                        ],
                        limit=1
                    )
                    
                    if not data:
                        continue
                        
                    current = data[0]
                    
                    # Get 20-day average price
                    hist_data = broker.get_stock_data(
                        ts_code,
                        trade_date,
                        fields=['daily.close'],
                        limit=20
                    )
                    
                    if not hist_data or len(hist_data) < 20:
                        continue
                        
                    avg_price = np.mean([d['daily.close'] for d in hist_data])
                    
                    # Calculate factors
                    risk_penalty = self.compute_risk_penalty(
                        volatility=current.get('factors.volat_20'),
                        turnover=current.get('daily_basic.turnover_rate'),
                        price=current.get('daily.close'),
                        avg_price=avg_price
                    )
                    
                    val_multiscore = self.compute_val_multiscore(
                        pe=current.get('daily_basic.pe_ttm'),
                        pb=current.get('daily_basic.pb'),
                        ps=current.get('daily_basic.ps'),
                        dv=current.get('daily_basic.dv_ratio')
                    )
                    
                    values.append((
                        ts_code,
                        trade_date,
                        now,
                        risk_penalty,
                        val_multiscore
                    ))
                    
                except Exception as e:
                    LOGGER.error(
                        "Error processing stock: %s error=%s",
                        ts_code,
                        str(e),
                        exc_info=True,
                        extra=LOG_EXTRA
                    )
                    continue
                    
            if values:
                try:
                    with broker.db.write_session() as session:
                        session.executemany(sql, values)
                    processed += len(values)
                except Exception as e:
                    LOGGER.error(
                        "Error saving batch results: %s",
                        str(e),
                        exc_info=True,
                        extra=LOG_EXTRA
                    )
                    
        LOGGER.info(
            "Batch processing completed: processed %d/%d stocks",
            processed,
            len(ts_codes),
            extra=LOG_EXTRA
        )